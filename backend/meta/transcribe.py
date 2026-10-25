"""Budgeted Muse realtime protocol. Credentials authenticate only in handshake."""
import asyncio
from contextlib import asynccontextmanager
import json
import os
import uuid
import ssl
import certifi
import websockets
from backend.meta.budget import AudioLease, MAX_AUDIO_SECONDS
from backend.meta.intent import safe_error

URI="wss://api.meta.ai/v1/asr/realtime"

def handshake():
    return {"authorization":{"accessToken":"Bearer "+os.environ["MODEL_API_KEY"]},
        "audioEncoding":"PCM_24KHZ","model":os.getenv("META_ASR_MODEL","muse-voice-transcribe-1.0"),
        # Omit languageBias: Muse detects the language, including code-switching.
        # Do not bias experimental Russian speech toward English instead.
        "mode":"ENDPOINTING","partialMode":"CUMULATIVE","emitAudioProgress":True,
        # Documented vocabulary hints, not an expected transcript or demo order.
        # Keep recognition multilingual and include both protect/reveal vocabulary.
        "keywords":["模糊","取消模糊","我的脸","blur","unblur",
                    "размытие","снимай","моё лицо","desenfoque","difumina","mi cara",
                    "Blur my face","请取消我脸上的模糊","请帮我去掉模糊","不要取消模糊",
                    "Не снимай меня","Quita el desenfoque de mi cara","Don't unblur me","No quites el desenfoque"]}

class MeteredSocket:
    def __init__(self,ws,lease):
        self.ws=ws; self.lease=lease

    async def send(self,payload):
        if isinstance(payload,bytes):
            self.lease.before_send(payload)
        async with asyncio.timeout(5):
            await self.ws.send(payload)

    def __aiter__(self):
        return self._events()

    async def _events(self):
        async for message in self.ws:
            if isinstance(message,str):
                try:self.lease.observe(json.loads(message))
                except ValueError:pass
            yield message

@asynccontextmanager
async def connection(max_seconds=MAX_AUDIO_SECONDS,budget=None):
    lease=AudioLease(budget=budget,max_seconds=max_seconds)
    ws=None; clean=False
    try:
        lease.check()
        ws=await websockets.connect(URI+"?sessionId="+str(uuid.uuid4()),open_timeout=15,close_timeout=2,max_size=1_000_000,ssl=ssl.create_default_context(cafile=certifi.where()))
        async with asyncio.timeout(15):
            await ws.send(json.dumps(handshake()))
            response=json.loads(await ws.recv())
        if "sessionId" not in response:
            raise RuntimeError(safe_error(response.get("message","Muse handshake rejected")))
        async with asyncio.timeout(lease.max_seconds+30):
            yield MeteredSocket(ws,lease),response["sessionId"]
        clean=ws.close_code==1000
    finally:
        if ws is not None:
            await ws.close()
        lease.close(clean=clean)

async def probe():
    """Small actual streaming request; handshake and processing aren't transcript proof."""
    messages=[]
    async with connection(max_seconds=2) as (ws,session_id):
        for _ in range(4):
            await ws.send(bytes(3840))  # 80 ms of PCM silence, real-time paced
            await asyncio.sleep(.08)
        await ws.send(json.dumps({"type":"endStream"}))
        async with asyncio.timeout(15):
            async for payload in ws:
                event=json.loads(payload)
                if event.get("type")=="error":
                    raise RuntimeError(safe_error(event.get("message","Muse error")))
                messages.append(event.get("type","unknown"))
    return {"status":"STREAM_ACCEPTED","provider_session_id":session_id,"events":messages,"note":"Silence probe; no microphone transcript demonstrated"}
