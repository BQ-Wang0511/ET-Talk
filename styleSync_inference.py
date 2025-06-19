from os import listdir, path
import numpy as np
import librosa
import scipy, cv2, os, sys, argparse #, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
import platform
from whisper.audio2feature import Audio2Feature
from PIL import Image
from data_utils.blending import get_image
import torchvision.transforms as transforms
from skimage import transform as trans
from models import ETTalk

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--ckpt', type=str, 
                    help='Name of saved checkpoint to load weights from', required=True)


parser.add_argument('--face', type=str, 
                    help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
                    help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
                                default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
                    help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
                    default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
                    'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')
parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                    'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')
parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='Prevent smoothing face detections over a short temporal window')


parser.add_argument('--face_det_batch_size', type=int, 
                    help='Batch size for face detection', default=8)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=8)

parser.add_argument('--resize_factor', default=1, type=int, 
            help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')


parser.add_argument('--image_size', type=int, default=256, help='Image size for the model.')
parser.add_argument('--resize', action='store_true', help='Resize the input image to 1/4 of the original size.')
parser.add_argument('--noparse', action='store_true', help='Enable face parsing.')
parser.add_argument('--acc', action='store_true', help='audio encoded with acc when combine with video')
parser.add_argument('--eval', action='store_true', help='evaluate on ckpt dir')
parser.add_argument('--save_sample', action='store_true', help='save sample image')

parser.add_argument('--mask_ratio', help='mask ratio', default=0.4, type=float)
parser.add_argument('--crop_down', help='crop down', default=0.1, type=float)

parser.add_argument('--no_pre_bbox', action='store_true', help='no pre bbox')
parser.add_argument('--ref_num', help='reference image num', default=1, type=int)

parser.add_argument('--audio_type',default=1, type=int, help='audio type')
parser.add_argument('--tmp_dir',default=None, type=str, help='temp dir to store video')

args = parser.parse_args()

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    args.static = True

def get_smoothened_boxes(boxes, T):
    new_boxes = []
    for i in range(len(boxes)):
        min_dix = max(0, i-T)
        max_dix = min(len(boxes), i+T)
        window = boxes[min_dix:max_dix]
        smooth_bbox = np.mean(window, axis=0).astype(np.int32)
        new_boxes.append(smooth_bbox)
    return new_boxes


def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=device)

    batch_size = args.face_det_batch_size
 
    resized_images = []
    if args.resize:
        resized_images = [cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4)) for image in images]
    else:
        resized_images = images

    while 1:
        predictions = []
        try:
            
            for i in tqdm(range(0, len(resized_images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(resized_images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite(f'{args.tmp_dir}/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        if args.resize:
            rect = [coord * 4 for coord in rect]  # Scale up the detection coordinates by 4
        
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth : boxes = get_smoothened_boxes(boxes, T=2)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    #torch.cuda.empty_cache()
    return results 


def datagen(frames, audios):
    # double the frames, from [0, N) to [0, N) + (N, 0]
   
    frames = frames + frames[::-1]
    
    img_batch, audio_batch, frame_batch, coords_batch, metric_batch = [], [], [], [], []
    
    if args.box[0] == -1:
        if not args.static:
            s3fd_dir=os.path.join(os.path.dirname(args.face),'s3fd_coords')
            if not os.path.exists(s3fd_dir) and not args.no_pre_bbox:
                os.makedirs(s3fd_dir)
            coords_path=os.path.join(s3fd_dir,os.path.basename(args.face).split('.')[0]+'.npy')
            if os.path.exists(coords_path) and not args.no_pre_bbox:
                print('Using the specified bounding box instead of face detection...')
                coords_list=np.load(coords_path)
                if args.crop_down!=0:
                    for i in range(len(coords_list)):
                        y1, y2, x1, x2 = coords_list[i]
                        h=y2-y1
                        y1+=h*args.crop_down
                        y2+=h*args.crop_down
                        coords_list[i]=y1, y2, x1, x2
                face_det_results=[[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f,(y1, y2, x1, x2) in zip(frames,coords_list)]
            else:
                face_det_results = face_detect(frames)
                coords_list=[c[1] for c in face_det_results]
                if not args.no_pre_bbox :
                    np.save(coords_path,coords_list)
                if args.crop_down!=0:
                    for i in range(len(face_det_results)):
                        y1, y2, x1, x2 = coords_list[i]
                        h=y2-y1
                        y1+=int(h*args.crop_down)
                        y2+=int(h*args.crop_down)
                        face_det_results[i][0]=frames[i][y1:y2,x1:x2]
                        face_det_results[i][1]=y1, y2, x1, x2
               
        else:
            face_det_results = face_detect([frames[0]])
            if args.crop_down!=0:
                for i in range(len(face_det_results)):
                    y1, y2, x1, x2 = face_det_results[i][1]
                    h=y2-y1
                    y1+=int(h*args.crop_down)
                    y2+=int(h*args.crop_down)
                    face_det_results[i][0]=frames[i][y1:y2,x1:x2]
                    face_det_results[i][1]=y1, y2, x1, x2
            
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
    

    for i, a in enumerate(audios):
        idx = 0 if args.static else i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()
        #生成argss.ref_num个0——len(frames)的随机数(整数不重复)
        
        
        face = cv2.resize(face, (args.image_size, args.image_size))
                
        img_batch.append(face)
        audio_batch.append(a)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)
        

        if len(img_batch) >= args.wav2lip_batch_size:
    
            img_batch, audio_batch = np.asarray(img_batch), torch.stack(audio_batch)

            img_masked = img_batch[:,:,:,args.ref_num*3-3:args.ref_num*3].copy()

            
            img_masked[:, int(args.image_size*args.mask_ratio):] = 0
           
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.

            yield img_batch, audio_batch, frame_batch, coords_batch, metric_batch
            img_batch, audio_batch, frame_batch, coords_batch, metric_batch = [], [], [], [], []

    if len(img_batch) > 0:

        img_batch, audio_batch = np.asarray(img_batch), torch.stack(audio_batch)

        img_masked = img_batch[:,:,:,args.ref_num*3-3:args.ref_num*3].copy()
   
        img_masked[:, int(args.image_size*args.mask_ratio):] = 0

        img_batch = (np.concatenate((img_masked, img_batch), axis=3) / 255)

        yield img_batch, audio_batch, frame_batch, coords_batch, metric_batch

audio_step_size = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    if device == 'cuda:':
        checkpoint = torch.load(checkpoint_path,map_location=torch.device('cuda'))
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    
    generator = ETTalk(
        args.image_size, 512, 8, channel_multiplier=2,ref_num=args.ref_num,audio_encoder_idx=args.audio_type
    ).to(device)
    #emb_g = EmbeddingGenerator().to(device)
    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g"])
    
    return generator.eval()


def main():
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)
    # if args.crop_down!=0:
    #     args.no_pre_bbox=True
    print ("Number of frames available for inference: "+str(len(full_frames)))
    audio_name = args.audio.split('/')[-1].split('.')[0]
    video_name=args.face.split('/')[-1].split('.')[0]
    ckpt_dir=os.path.dirname(args.ckpt)
    
    if args.eval:
        args.outfile=f'{ckpt_dir}/{video_name}_{audio_name}.mp4'
        print(f'outfile: {args.outfile}')

    if args.tmp_dir is None and args.eval:
        args.tmp_dir=f'{ckpt_dir}/temp'
    elif args.tmp_dir is None:
        args.tmp_dir=os.path.join(os.path.dirname(args.outfile),'temp')
        #ckpt_dataset_name=args.ckpt.split('-')[2]
    os.makedirs(args.tmp_dir, exist_ok=True)

    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        #audio_name=args.audio.split('/')[-1].split('.')[0]
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, f'{args.tmp_dir}/{audio_name}.wav')

        subprocess.call(command, shell=True)
        args.audio = f'{args.tmp_dir}/{audio_name}.wav'


 
    
    if args.audio_type==1:
        whisper_processor = Audio2Feature(model_path='tiny')
    else:
        whisper_processor = Audio2Feature(model_path='/data/wangbaiqin/project/MuseTalk/models/whisper/large-v3.pt',whisper_model_type='large-v3')
    whisper_feature = whisper_processor.audio2feat(args.audio)
    print("Whisper feature shape: {}".format(whisper_feature.shape))
    # if args.audio_type==3:
    #     whisper_feature=whisper_feature[:,-1]
    audio_chunks = whisper_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
    if args.audio_type==3:
        audio_chunks = [torch.tensor(audio_chunk[::audio_chunk.shape[0]//10]).unsqueeze(0).float() for audio_chunk in audio_chunks]
    else:
        audio_chunks = [torch.tensor(audio_chunk).unsqueeze(0).float() for audio_chunk in audio_chunks]
    

    print("Length of audio chunks: {}".format(len(audio_chunks)))
    print("Shape of audio chunks: {}".format(audio_chunks[0].shape))

    full_frames = full_frames[:len(audio_chunks)]

    batch_size = args.wav2lip_batch_size
    zero_timestep = torch.zeros([])
    gen = datagen(full_frames.copy(), audio_chunks)
    
    for i, (img_batch, audio_batch, frames, coords, metric) in enumerate(tqdm(gen, 
                                            total=int(np.ceil(float(len(audio_chunks))/batch_size)))):
        if i == 0:
            g= load_model(args.ckpt)
            print ("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter(f'{args.tmp_dir}/result.avi', 
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        audio_batch = audio_batch.to(device)

        if i == 0:
            ref_frame = img_batch[7]
        
            
        with torch.no_grad():
            pred_image = g(img_batch, audio_batch)
  
        pred = pred_image.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        
        for idx,(p, f, c) in enumerate(zip(pred, frames, coords)):
            
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            if args.noparse:
                f[y1:y2, x1:x2] = p
                out.write(f)
            else:
                c=x1,y1,x2,y2
                new_frame=get_image(f,p,c)
                out.write(new_frame)

    out.release()
    
        
    if args.acc:
        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, f'{args.tmp_dir}/result.avi', args.outfile)
    else:
        command="ffmpeg -y -v warning -i {} -i {} -hide_banner -strict -2 -q:v 1 -c:a libvorbis {}".format(args.audio,f'{args.tmp_dir}/result.avi',args.outfile)
    
    subprocess.call(command, shell=platform.system() != 'Windows')
     
    print("output video saved at: {}".format(args.outfile))
    
if __name__ == '__main__':
    main()
