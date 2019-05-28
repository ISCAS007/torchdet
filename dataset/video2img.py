#for f in `ls $1`
#do
#    echo "convert $f to image"
#    echo "ffmpeg -i $f -vf fps=1 $2"
#done

import os
import glob

class video2img:
    def __init__(self,input_root_dir,output_root_dir,fps=1):
        self.input_root_dir=input_root_dir
        self.output_root_dir=output_root_dir
        self.fps=fps
        
        if input_root_dir.find('CVPRLab')>=0:
            files=glob.glob(os.path.join(input_root_dir,'*'),recursive=True)
        elif input_root_dir.find('VisiFire')>=0:
            # remove Car_Counting.wmv and barbeqraw.avi
            files=glob.glob(os.path.join(input_root_dir,'*','*'),recursive=True)
        elif input_root_dir.find('FireSense')>=0:
            files=glob.glob(os.path.join(input_root_dir,'**','*'),recursive=True)
        else:
            assert False
        video_suffix=('avi','mp4','wmv')
        self.videos=[f for f in files if f.lower().endswith(video_suffix)]
        
        self.videos.sort()
        assert len(self.videos)>0
        print('find {} video'.format(len(self.videos)))
        print(self.videos)
        
    def convert2img(self):
        for idx,video_file in enumerate(self.videos):
            label=self.convert2label(video_file)
            output_format=os.path.join(os.path.dirname(video_file),'_'.join(label+[str(idx),'%04d.jpg']))
            output_format=output_format.replace(self.input_root_dir,self.output_root_dir)
            os.makedirs(os.path.dirname(output_format),exist_ok=True)
            cmd='ffmpeg -i {} -vf fps={} {}'.format(video_file,self.fps,output_format)
            print(cmd)
            os.system(cmd)
            
    def convert2label(self,video_file):
        label=[]
        if video_file.find('CVPRLab')>=0:
            """
            wildfire contain only smoke
            """
            basename=os.path.basename(video_file)
            if basename.find('smoke_or_flame_like_object')>=0:
                label+=['normal']
            else:
                if basename.find('flame')>=0 or \
                basename.find('heptane')>=0 or \
                basename.find('gasoline')>=0:
                    label+=['fire']
                
                if basename.find('smoke')>=0:
                    label+=['smoke']
        elif video_file.find('VisiFire')>=0:
            if video_file.find('FireClips')>=0:
                label+=['fire']
            elif video_file.find('ForestSmoke')>=0 or video_file.find('SmokeClips')>=0:
                label+=['smoke']
            else:
                assert False,'video_file={}'.format(video_file)
        elif video_file.find('FireSense')>=0:
            if video_file.find('FireSense/fire/pos')>=0:
                label+=['fire']
            elif video_file.find('FireSense/smoke/pos')>=0:
                label+=['smoke']
            else:
                assert video_file.find('neg')>=0
                label+=['normal']
        else:
            assert False
            
        return label
            
if __name__ == '__main__':
    #in_dir='dataset/smoke/CVPRLab'
    #in_dir='dataset/smoke/VisiFire'
    in_dir='dataset/smoke/FireSense'
    out_dir=in_dir+'_img'
    v2i=video2img(in_dir,out_dir)
    v2i.convert2img()