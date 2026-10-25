import {useEffect,useRef,useState} from 'react';

// Pull protected JPEGs from the existing worker. This recovers when a backend
// restart closes Chrome's long-lived MJPEG image without a useful error event.
export default function ProtectedPreview(){
 const canvas=useRef<HTMLCanvasElement>(null);
 const [ready,setReady]=useState(false);
 useEffect(()=>{
   let disposed=false,timer:ReturnType<typeof setTimeout>,controller:AbortController|undefined;
   async function update(){
     controller=new AbortController();
     const timeout=setTimeout(()=>controller?.abort(),2000);
     try{
       const response=await fetch('/api/preview/protected.jpg',{cache:'no-store',signal:controller.signal});
       if(!response.ok)throw new Error('Preview unavailable');
       const bitmap=await createImageBitmap(await response.blob());
       if(!disposed&&canvas.current){
         const target=canvas.current;
         if(target.width!==bitmap.width||target.height!==bitmap.height){target.width=bitmap.width;target.height=bitmap.height}
         target.getContext('2d')?.drawImage(bitmap,0,0);
         setReady(true);
       }
       bitmap.close();
     }catch{if(!disposed)setReady(false)}
     finally{clearTimeout(timeout);if(!disposed)timer=setTimeout(update,60)}
   }
   update();
   return()=>{disposed=true;clearTimeout(timer);controller?.abort()};
 },[]);
 return <><canvas ref={canvas} className="video" role="img" aria-label="Live protected video; protection is rendered in the actual image pixels"/>{!ready&&<div className="preview-reconnecting" role="status">Reconnecting protected preview…</div>}</>
}
