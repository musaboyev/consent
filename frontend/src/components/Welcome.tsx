import {memo,useEffect,useRef,useState,type RefObject} from 'react';
export default memo(function Welcome({onEnter,onBegin,headerLogo}:{onEnter:()=>void;onBegin:()=>void;headerLogo:RefObject<HTMLImageElement|null>}){
 const [entering,setEntering]=useState(false),[flight,setFlight]=useState<React.CSSProperties>();
 const logo=useRef<HTMLImageElement>(null),timer=useRef<ReturnType<typeof setTimeout>>(undefined);
 useEffect(()=>()=>clearTimeout(timer.current),[]);
 const enter=()=>{
  if(entering)return;
  const from=logo.current?.getBoundingClientRect(),to=headerLogo.current?.getBoundingClientRect();
  const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
  if(from&&to&&!reduced)setFlight({'--move-x':`${to.left-from.left}px`,'--move-y':`${to.top-from.top}px`,'--scale':to.width/from.width} as React.CSSProperties);
  setEntering(true);onBegin();timer.current=setTimeout(onEnter,reduced?0:720);
 };
 return <section className={`welcome ${entering?'entering':''}`} aria-label="Welcome to Consent">
  <div className="entry-ripple"/>
  <button className="welcome-button" onClick={enter} aria-label="Enter Consent live session">
   <span className="logo-flight" style={flight}><img ref={logo} className="welcome-logo" src="/consent-wordmark.svg" alt="Consent"/></span>
   <span className="welcome-attribution">Powered by Meta AI</span>
   <span className="enter-hint">Touch to enter</span>
  </button>
  <p className="welcome-caption">Your visibility. Your choice.</p>
 </section>
});
