
import os
import sounddevice as sd
import soundfile as sf
    
def play_chime(sound_path_input='speech_on'):
    sound_paths = {
        'sound_path_1': r'C:\Windows\Media\chimes.wav',
        'sound_path_2': r'C:\Windows\Media\ding.wav',
        'speech_off': r'C:\Windows\Media\Speech Off.wav',
        'speech_dis': r'C:\Windows\Media\Speech Disambiguation.wav',
        'speech_on': r'C:\Windows\Media\Speech On.wav'
    }
    sound_path = sound_paths.get(sound_path_input)
    if not sound_path or not os.path.exists(sound_path):
        # No path or file missing; don't raise to avoid breaking the flow
        print(f"Sound path for '{sound_path_input}' not found or file missing: {sound_path}")
        return

    # Preferred: use sounddevice + soundfile if available (non-blocking)
    try:
        if sd is not None and sf is not None:
            data, sr = sf.read(sound_path, dtype='float32')
            sd.play(data, sr)  # non-blocking
            return
    except Exception:
        # swallow error and try other backends
        pass

    # Windows fallback: winsound (async)
    try:
        if os.name == 'nt':
            winsound.PlaySound(sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            return
    except Exception:
        pass

    # Unix/MacOS fallback: try common command-line players
    try:
        for cmd in ('afplay', 'aplay', 'paplay'):
            if shutil.which(cmd):
                # Launch non-blocking
                subprocess.Popen([cmd, sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
    except Exception:
        pass

    # Last resort: notify but don't raise
    print(f"Unable to play chime for '{sound_path_input}'.")

