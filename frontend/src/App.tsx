import {useCallback,useEffect,useRef,useState} from 'react';
import {command,type State,type Recording} from './api';
import ProtectedPreview from './components/ProtectedPreview';
import Icon from './components/Icon';
import Welcome from './components/Welcome';

const clock=(ms:number)=>`${Math.floor(ms/60000).toString().padStart(2,'0')}:${Math.floor(ms/1000%60).toString().padStart(2,'0')}`;
const person=(id:string)=>id.replace('p_','Person ');
const actionLabels:Record<string,string>={PROTECT_PERSON:'Blur requested',UNPROTECT_PERSON:'Reveal requested',NO_ACTION:'No privacy request',NEEDS_CLARIFICATION:'Clarification needed'};
const eventLabels:Record<string,string>={PERSON_SEEN:'Person recognized',PROTECTION_REMOVED:'Face revealed by request',PROTECTION_ACTIVATED:'Face protection activated',PERSON_LEFT:'Person left view',PERSON_REIDENTIFIED:'Remembered person returned',VOICE_REQUEST:'Spoken request received',OUTPUT_SHIELDED:'Output protected during uncertainty',OUTPUT_RESUMED:'Targeted output resumed',RECORDING_STARTED:'Recording started',RECORDING_STOPPED:'Recording finalized'};

function RecordingPair({recording:r,index}:{recording:Recording;index:number}){
 const [open,setOpen]=useState(false),[playing,setPlaying]=useState<'original'|'protected'|null>(null);
 const [availability,setAvailability]=useState<Record<string,string>>({}),[mediaError,setMediaError]=useState('');
 const url=`/api/recordings/${r.recording_id}`;
 useEffect(()=>{
  if(!open||r.status!=='FINALIZED')return;
  const controller=new AbortController();
  for(const kind of ['original','protected']){
   fetch(`${url}/${kind}`,{method:'HEAD',cache:'no-store',signal:controller.signal})
    .then(response=>setAvailability(v=>({...v,[kind]:response.ok?'available':`Unavailable (HTTP ${response.status})`})))
    .catch(e=>{if(e.name!=='AbortError')setAvailability(v=>({...v,[kind]:'Availability check failed — reopen to retry'}))});
  }
  return()=>controller.abort();
 },[open,r.status,url]);
 return <details className="recording-pair" onToggle={e=>{setOpen(e.currentTarget.open);if(!e.currentTarget.open)setPlaying(null)}}><summary><span>Recording {String(index).padStart(2,'0')}</span><span className="recording-meta">{clock(r.duration_ms||0)} · {r.status.toLowerCase()}</span></summary>
  {r.status==='FINALIZED'?<div className="recording-content">
   {(['protected','original'] as const).map(kind=><div className="recording-file" key={kind}>
    <div><strong>{kind==='protected'?'Protected':'Original / unprotected'}</strong><small>{kind==='original'?'Developer diagnostic · contains unprotected faces':'Live privacy processing applied'} · {clock(r.duration_ms||0)} · video only</small></div>
    <div className="file-actions"><button disabled={availability[kind]!=='available'} onClick={()=>{setMediaError('');setPlaying(kind)}} aria-label={`Play ${kind} recording ${index}`}><Icon name="play"/>Play</button>
     {availability[kind]==='available'?<a href={`${url}/${kind}`} download={`${r.session_id||'session'}_${r.recording_id}_${kind}.mp4`} aria-label={`Download ${kind} recording ${index}`}><Icon name="download"/>Download</a>:<span className="muted">{availability[kind]||'Checking file…'}</span>}
    </div>
   </div>)}
   {playing&&<div className="recording-player"><p>{playing==='original'?'Original / unprotected · developer diagnostic':'Protected recording'}</p><video key={playing} controls preload="metadata" src={`${url}/${playing}`} aria-label={`${playing==='original'?'Developer diagnostic original':'Protected'} recording ${index}`} onError={()=>setMediaError(`${playing==='original'?'Original':'Protected'} video could not be played. The file may be missing or damaged.`)}/></div>}
   {mediaError&&<p className="provider-error" role="alert">{mediaError}</p>}
  </div>:<p className="muted">{r.status==='RECORDING'?'Both outputs are being encoded live. Stop recording to finalize playback.':`Recording ${r.status.toLowerCase()}. Neither file is advertised as finalized.`}</p>}
 </details>
}

export default function App(){
 const [state,setState]=useState<State|null>(null),[connected,setConnected]=useState(false),[error,setError]=useState(''),[notice,setNotice]=useState(''),[busy,setBusy]=useState(false);
 const [transitioning,setTransitioning]=useState(false);
 const [entered,setEntered]=useState(false),[tab,setTab]=useState<'live'|'recordings'>('live'),[source,setSource]=useState('iphone');
 const headerLogo=useRef<HTMLImageElement>(null),heading=useRef<HTMLHeadingElement>(null),settings=useRef<HTMLDialogElement>(null),reset=useRef<HTMLDialogElement>(null);
 useEffect(()=>{
  let socket:WebSocket|undefined,retry:ReturnType<typeof setTimeout>,disposed=false;
  const connect=()=>{
   if(disposed)return;
   socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/api/events`);
   socket.onopen=()=>setConnected(true);
   socket.onmessage=(event)=>{try{setState(JSON.parse(event.data))}catch{setError('Invalid session data')}};
   socket.onerror=()=>socket?.close();
   socket.onclose=()=>{setConnected(false);if(!disposed)retry=setTimeout(connect,1500)};
  };connect();
  return()=>{disposed=true;clearTimeout(retry);socket?.close()};
 },[]);
 useEffect(()=>{const update=()=>{document.documentElement.dataset.hidden=String(document.hidden)};update();document.addEventListener('visibilitychange',update);return()=>document.removeEventListener('visibilitychange',update)},[]);
 useEffect(()=>{if(entered)heading.current?.focus()},[entered]);
 const run=async(path:string)=>{setBusy(true);setError('');try{await command(path)}catch(e){setError((e as Error).message)}finally{setBusy(false)}};
 const copyPhoneToken=async()=>{try{const response=await fetch('/api/source/iphone/pair');if(!response.ok)throw new Error('Phone pairing unavailable');const data=await response.json();await navigator.clipboard.writeText(data.upload_token);setNotice('Upload token copied. Paste it into the iPhone bridge. This is not a Meta API key.')}catch(e){setNotice((e as Error).message)}};
 const waitingPhone=state?.status==='WAITING_SOURCE';
 const sourceRecovering=['RECOVERING_SOURCE','SOURCE_ERROR'].includes(state?.status||'');
 const active=state?.status==='RUNNING'||sourceRecovering||waitingPhone,starting=state?.status==='STARTING';
 const sourceError=state?.output_state==='ERROR'||sourceRecovering;
 const live=connected&&state?.status==='RUNNING'&&!sourceError;
 const voiceOn=['CONNECTING','CONNECTED','LISTENING','TRANSCRIBING'].includes(state?.voice?.status||'');
 const budget=state?.voice?.budget,voice=state?.voice;
 const latest=[...(state?.recordings||[])].reverse();
 const connectionLabel=!connected?'Engine disconnected':waitingPhone?'Waiting for iPhone':sourceError?(state?.reason?.toLowerCase().includes('disconnect')||sourceRecovering?'Camera disconnected':'Preview needs attention'):live?(state?.source==='iphone'?'iPhone connected':'Webcam connected'):starting?'Starting camera':'No active camera';
 const micLabel=!connected?'Engine disconnected':voice?.error?'Microphone needs attention':voice?.status==='CONNECTING'?'Connecting microphone':voiceOn?(voice?.transcript_final===false?'Hearing your request…':'Microphone on'):'Microphone off';
 const action=voice?.intent?.intent;
 const metric=(key:string,suffix='')=>state?.metrics[key]!=null?`${state.metrics[key]}${suffix}`:'—';
 const enter=useCallback(()=>setEntered(true),[]);
 const beginEntry=useCallback(()=>setTransitioning(true),[]);
 return <>
  <div className={`liquid-background ${entered?'quiet-motion':''}`} aria-hidden="true"><div className="liquid-contour"/></div>
  <div className={`dashboard ${entered?'is-entered':transitioning?'is-entering':''}`} inert={!entered}>
   <header className="topbar"><div className="header-inner">
    <button className="brand" onClick={()=>setTab('live')} aria-label="Consent live session"><img ref={headerLogo} src="/consent-wordmark.svg" alt="Consent"/><span>Powered by Meta AI</span></button>
    <nav aria-label="Main navigation"><button className={tab==='live'?'selected':''} aria-current={tab==='live'?'page':undefined} onClick={()=>setTab('live')}>Live session</button><button className={tab==='recordings'?'selected':''} aria-current={tab==='recordings'?'page':undefined} onClick={()=>setTab('recordings')}>Recordings</button></nav>
    <div className="header-status"><button className="icon-button settings-button" aria-label="Open settings and diagnostics" onClick={()=>settings.current?.showModal()}><Icon name="settings"/></button><span className="connection"><i className={`dot ${live?'live':'muted-dot'}`}/>{connectionLabel}</span></div>
   </div></header>
   <main className="shell">
    <div className="page-heading"><div><p className="eyebrow">{tab==='live'?'LIVE SESSION':'PAIRED RECORDINGS'}</p><h1 ref={heading} tabIndex={-1}>{tab==='live'?'Your visibility. Your choice.':'Every moment, protected.'}</h1></div>
     <div className="record-actions">{!active&&!starting&&<select aria-label="Camera source" value={source} onChange={e=>setSource(e.target.value)}><option value="iphone">iPhone · Meta SDK</option><option value="webcam">Webcam · fallback</option></select>}
      <button className={`record-button ${state?.recording?'is-recording':''}`} disabled={busy||!connected||(!state?.recording&&(starting||waitingPhone||sourceError))} onClick={()=>run(active?'recording/'+(state?.recording?'stop':'start'):'session/start?source='+source)}>{state?.recording&&<i className="dot recording"/>}{starting?'Preparing camera…':waitingPhone?'Waiting for iPhone…':sourceError&&!state?.recording?'Camera unavailable':active?(state?.recording?'Stop & finalize':'Start recording'):'Start camera session'}</button>
     </div>
    </div>
    {(error||state?.error)&&<div className="alert" role="alert">{error||state?.error}</div>}
    <div className={`workspace ${tab==='recordings'?'recordings-view':''}`}>
     <div className="live-column" hidden={tab!=='live'}>
      <section className="preview-panel panel" aria-label="Protected live preview">
       <div className="preview-heading"><strong>Protected preview</strong><span><i className={`dot ${live?'live':''}`}/>{live?'Live':sourceError?'Preview unavailable':starting?'Starting':'Waiting'}</span>{state?.recording&&<span className="recording-clock"><i className="dot recording"/>{clock(state.recording.duration_ms||0)}</span>}</div>
       <div className="video-wrap"><ProtectedPreview/>
        {(!active||!connected||waitingPhone||starting)&&<div className="waiting"><Icon name="eye"/><h2>{waitingPhone?'Connect your iPhone':starting?'Preparing the camera…':!connected?'Connecting to the local engine…':'Your session starts here.'}</h2><p>{waitingPhone?'Open settings for the iPhone bridge setup.':starting?'The preview will appear when ready.':!connected?'Reconnecting does not clear privacy preferences.':'Choose your camera and start a session.'}</p></div>}
        {active&&connected&&state?.output_state!=='NORMAL'&&<div className="shield-label" role="status">{state?.reason||'Protection active while the camera resolves uncertainty'}</div>}
        <div className="video-footer"><span>{state?.recording?'RECORDING BOTH OUTPUTS':'PROTECTED PREVIEW'}</span><span>{clock(state?.timestamp_ms||0)}</span></div>
       </div>
      </section>
      <section className="participant-strip panel" aria-label="People and remembered privacy preferences">
       {!state?.tracks.length?<p className="empty">People appear here when detected. No request does not mean consent.</p>:state.tracks.map(p=><div className={`person ${p.visibility!=='VISIBLE'?'absent':''}`} key={p.person_id}>
        <span className="avatar"><Icon name="person"/></span><div className="person-name"><strong>{p.person_id.startsWith('p_')?person(p.person_id):'Person · matching'}</strong><small>{p.visibility!=='VISIBLE'?'Out of frame · preference remembered':p.identity_status!=='KNOWN'?'Identity resolving':'In view'}</small>{p.gesture_progress>0&&p.visibility==='VISIBLE'&&<progress max="1" value={p.gesture_progress} aria-label={`${person(p.person_id)} gesture hold`}/>}</div>
        <span className={`pill ${p.privacy==='BLURRED'||p.temporary_protection?'protected':''}`}>{p.temporary_protection?'Matching · protected':p.privacy==='BLURRED'?'Protected':p.privacy==='PENDING'?'Matching':p.visibility==='VISIBLE'?'Visible':'Unblurred'}</span>
       </div>)}
      </section>
     </div>
     <aside className="privacy-panel panel" hidden={tab!=='live'}>
      <h2>Privacy controls</h2>
      <div className="gesture-instructions"><div><Icon name="eye"/><div><h3>Cover eyes</h3><p>Hold to blur your face</p></div></div><div><Icon name="peace"/><div><h3>Peace sign</h3><p>Reveal your face</p></div></div><div><Icon name="mic"/><div><h3>Use your voice</h3><p>Say “Blur me” or “Unblur me”</p></div></div></div>
      <div className="latest-voice"><h3>Latest voice request</h3><p className={`transcript ${!voice?.transcript?'muted':''}`} dir="auto">{voice?.transcript?`“${voice.transcript}”`:'Your spoken request will appear here.'}</p>
       {voice?.transcript_final===false?<p className="intent-state">Still transcribing · no action yet</p>:action&&<p className="intent-state"><i className="dot blue-dot"/><span className="sr-only">Latest interpreted action: </span>{actionLabels[action]||action}{voice?.target_person_id?` · ${person(voice.target_person_id)}`:''}</p>}
       {voice?.request_status&&<p className="request-status">{voice.request_status.replaceAll('_',' ').toLowerCase()}</p>}
       {(voice?.error||voice?.spark_error)&&<p className="provider-error" role="status">{voice.error||voice.spark_error}</p>}
      </div>
      <div className="microphone"><div className="mic-status"><i className={`dot ${voiceOn&&connected?'live':''}`}/>{micLabel}</div>{voiceOn&&<meter min="-80" max="0" value={Math.max(-80,voice?.microphone_peak_dbfs??-80)} aria-label="Mac microphone input level"/>}
       <button className="blue-button" disabled={busy||!active||!connected||(!voiceOn&&budget?.meta_disabled)} onClick={()=>run('voice/'+(voiceOn?'stop':'start'))}><Icon name="mic"/>{voiceOn?'Stop microphone':'Enable Meta voice'}</button>
       {budget?.meta_disabled&&<p className="provider-error">{budget.reason||'Meta features disabled by the application budget.'} Gestures remain available.</p>}
       <p className="footnote">Voice uses the Mac microphone. One visible person only; speaker labels do not identify faces. Uncertain requests cannot reveal a protected face.</p>
      </div>
     </aside>
     <section className="recordings-panel panel" aria-label="Paired recordings"><div className="section-heading"><h2>Recordings</h2><span>Original + protected</span>{tab==='live'&&latest.length>2&&<button className="text-button" onClick={()=>setTab('recordings')}>View all ({latest.length})</button>}</div>{state?.recording&&<p className="recording-notice"><i className="dot recording"/>Recording both outputs · {clock(state.recording.duration_ms||0)}</p>}
      {!latest.length?<p className="empty">Start recording to save both outputs. Stop finalizes playback.</p>:<div className={tab==='live'?'recording-list compact':'recording-list'}>{latest.map((r,i)=><RecordingPair key={r.recording_id} recording={r} index={latest.length-i}/>)}</div>}
      <p className="footnote recording-note">Encoded live. Video only. Original playback is a separate developer diagnostic action.</p>
     </section>
    </div>
    <p className="memory-caption">Preferences are remembered when you return.</p>
    <footer>Actual Meta SDK · simulated glasses · live iPhone camera. This local prototype does not enforce privacy in native Meta recordings.</footer>
   </main>
  </div>
  {!entered&&<Welcome headerLogo={headerLogo} onBegin={beginEntry} onEnter={enter}/>}
  <dialog ref={settings} className="settings-dialog" aria-labelledby="settings-title"><div className="dialog-heading"><h2 id="settings-title">Settings & diagnostics</h2><button className="icon-button" aria-label="Close settings" onClick={()=>settings.current?.close()}><Icon name="close"/></button></div>
   <p className="muted">The camera, privacy memory and recordings continue while this panel is open.</p>
   <details open><summary>Camera & session</summary><p>Source: {state?.source==='iphone'?'iPhone · Meta SDK / simulated glasses':state?.source==='webcam'?'Mac webcam · development fallback':'Not started'}</p><p>Status: {connectionLabel}</p><code>{state?.session_id||'No active session'}</code>
    {state?.status==='SOURCE_ERROR'&&state?.source==='webcam'&&<button disabled={busy} onClick={()=>run('source/retry')}>Retry camera · retain privacy memory</button>}
    <details><summary>iPhone bridge setup & recovery</summary><ol><li>Keep the Mac and iPhone on the same reachable network.</li><li>Start an iPhone session, then open ConsentLens Bridge on the phone.</li><li>Enter this Mac’s reachable network address with port <code>:8000</code>, not localhost. Copy the local upload token below and paste it into the bridge.</li><li>Tap Connect live iPhone camera. Allow Camera and Local Network access; keep the app in the foreground.</li></ol><p className="muted">If disconnected, reconnect the bridge. Existing privacy preferences are retained.</p><button onClick={copyPhoneToken}>Copy local phone upload token</button><p role="status">{notice}</p></details>
    {(active||state?.status==='ERROR')&&<button className="danger-button" disabled={busy} onClick={()=>reset.current?.showModal()}>End session & clear memory…</button>}
   </details>
   <details><summary>Meta voice & application budget</summary><p>Muse Voice: {voice?.status.toLowerCase()||'unavailable'} · Muse Spark: {voice?.spark_status?.toLowerCase()||'untested'}</p><p>Interpreter: {voice?.intent_route||'No request yet'}</p>{budget?<><strong className="budget-amount">{budget.remaining_estimated_usd==null?'Unavailable':`$${budget.remaining_estimated_usd.toFixed(6)}`}</strong><p>of ${budget.limit_usd.toFixed(2)} estimated available</p><meter min="0" max={budget.limit_usd} value={budget.remaining_estimated_usd||0} aria-label="Estimated Meta budget remaining"/><p className="footnote">Includes reservations; persists across restarts. Local safeguard, not an account-wide billing cap. Codex, AMD and local processing are excluded.</p>{(budget.uncertain_estimate_usd||0)>0&&<p className="footnote">${budget.uncertain_estimate_usd?.toFixed(6)} is estimated where complete provider usage was unavailable.</p>}{budget.meta_disabled&&<p className="provider-error">{budget.reason}</p>}</>:<p className="muted">Budget not available yet.</p>}<p className="footnote">Mandarin pronunciation remains sensitive. Displayed transcripts are the actual provider result. Text-only checks do not establish live speech accuracy.</p></details>
   <details><summary>Performance & event timeline</summary><div className="metrics"><div><strong>{metric('processed_fps')}</strong><span>processed FPS</span></div><div><strong>{metric('inference_ms_p95',' ms')}</strong><span>p95 vision</span></div><div><strong>{metric('capture_to_jpeg_age_ms_p95',' ms')}</strong><span>p95 admission → JPEG</span></div><div><strong>{metric('dropped_before_admission')}</strong><span>frames skipped</span></div></div><p className="footnote">Latest 300 processed frames. Excludes browser display; iPhone timing also excludes phone capture and network transit.</p><ol className="timeline">{[...(state?.events||[])].reverse().slice(0,20).map(e=><li key={e.event_id}><time>{clock(e.timestamp_ms)}</time><span>{eventLabels[e.type]||e.type.replaceAll('_',' ').toLowerCase()}</span><small>{e.person_id?person(e.person_id):''}</small></li>)}</ol></details>
  </dialog>
  <dialog ref={reset} className="confirm-dialog" aria-labelledby="reset-title"><h2 id="reset-title">End this session?</h2><p>This stops capture and clears everyone’s remembered privacy preferences. Starting again creates a new session.</p><p>To recover a disconnected phone, reconnect the bridge instead.</p><div className="confirm-actions"><button autoFocus onClick={()=>reset.current?.close()}>Keep session</button><button className="danger-button" disabled={busy} onClick={()=>{reset.current?.close();settings.current?.close();void run('session/end')}}>End & clear memory</button></div></dialog>
 </>
}
