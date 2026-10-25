import type {CSSProperties} from 'react';
type Name='eye'|'peace'|'mic'|'person'|'settings'|'download'|'close'|'play'|'stop';
export default function Icon({name,style}:{name:Name;style?:CSSProperties}){
 const paths:Record<Name,React.ReactNode>={
  eye:<><path d="M3 12s3-7 9-7 9 7 9 7-3 7-9 7-9-7-9-7Z"/><circle cx="12" cy="12" r="3"/><path d="m3 21 18-18"/></>,
  peace:<><path d="m9 13-2-8c-.7-3 2-4 3-1l2 7 2-8c.7-2.5 3-2 2.5.5L15 13l1-1c2-2 4 0 2.5 2l-1 4c-1 3-3 4-6 4-4 0-6-3-6-6v-4c0-2 3-2 3 0v3"/><path d="M8 13h4c3 0 4 3 1 4h-2l-1 2"/></>,
  mic:<><rect x="8" y="2" width="8" height="14" rx="4"/><path d="M5 10v3a7 7 0 0 0 14 0v-3M12 20v3m-4 0h8"/></>,
  person:<><circle cx="12" cy="8" r="4"/><path d="M4 22v-2a8 8 0 0 1 16 0v2"/></>,
  settings:<><path d="m10 3-1 3-3-1-2 3 2 2-1 3 3 1v3l3 1 2-2 3 1 2-3-2-2 1-3-3-1V4l-3-1-1 2Z"/><circle cx="11" cy="11" r="3"/></>,
  download:<><path d="M12 3v12m-4-4 4 4 4-4M4 16v5h16v-5"/></>,
  close:<path d="m6 6 12 12M6 18 18 6"/>,
  play:<path d="m8 4 12 8-12 8Z"/>,
  stop:<rect x="5" y="5" width="14" height="14" rx="1"/>
 };
 return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={style}>{paths[name]}</svg>
}
