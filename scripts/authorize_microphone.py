"""Main-thread local microphone preflight; no audio is saved or transmitted."""
import sounddevice as sd
import argparse
import time

parser=argparse.ArgumentParser()
parser.add_argument("--seconds",type=int,default=1)
args=parser.parse_args()
if not 1<=args.seconds<=60:raise SystemExit("Use 1–60 seconds")

print("Checking microphone. Allow Microphone for the launching app if macOS prompts.",flush=True)
try:
    print("Input device:",sd.query_devices(kind="input")["name"],flush=True)
    with sd.RawInputStream(samplerate=24000,channels=1,dtype="int16",blocksize=1920,latency="high") as stream:
        deadline=time.monotonic()+args.seconds
        blocks=0; overflows=0
        while time.monotonic()<deadline:
            samples,overflow=stream.read(1920)
            blocks+=1; overflows+=int(overflow)
        print(f"Microphone: 24 kHz mono, {blocks} blocks, {overflows} overflows, {stream.latency*1000:.1f} ms device buffer; device released.",flush=True)
except Exception as exc:
    raise SystemExit("Microphone unavailable: "+str(exc)+". Enable the launching app under System Settings > Privacy & Security > Microphone.")
