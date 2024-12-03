import webrtcvad

from utils.rtcvad_helpers import read_wave, vad_collector, write_wave, frame_generator

if __name__ == '__main__':
    data, sr = read_wave("samples/check_solo_10s.wav")
    
    vad = webrtcvad.Vad(3)
    
    frames = frame_generator(30, data, sr)
    frames = list(frames)
    
    segments = vad_collector(sr, 30, 300, vad, frames)
    
    #for i, segment in enumerate(segments):
        #path = 'tmp/chunk-%002d.wav' % (i,)
        #print(' Writing %s' % (path,))
        #write_wave(path, segment, sr)