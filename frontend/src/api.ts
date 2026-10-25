export type Track={person_id:string;privacy:string;visibility:string;identity_status:string;face_box:number[]|null;gesture_progress:number;protection_source:string|null;last_seen_ms:number;temporary_protection?:boolean};
export type PrivacyEvent={event_id:string;type:string;timestamp_ms:number;person_id:string|null;details:Record<string,unknown>};
export type Recording={recording_id:string;session_id?:string;status:string;frames:number;duration_ms:number};
export type Budget={limit_usd:number;estimated_consumed_usd?:number;reserved_usd?:number;remaining_estimated_usd:number|null;uncertain_estimate_usd?:number;meta_disabled:boolean;reason?:string|null;pricing_checked_at?:string};
export type State={session_id:string|null;status:string;source?:string;frame_id?:number;timestamp_ms?:number;tracks:Track[];events:PrivacyEvent[];metrics:Record<string,number|null>;output_state:string;protection_mode?:string;reason?:string|null;error?:string;recording?:Recording|null;recordings:Recording[];voice:{status:string;transcript?:string;transcript_final?:boolean;error?:string|null;spark_status?:string;spark_error?:string|null;intent_route?:string;real_transcripts?:number;assumption?:string;request_status?:string;target_person_id?:string|null;budget?:Budget;microphone_peak_dbfs?:number;asr_processed_ms?:number;captured_audio_ms?:number;sent_audio_ms?:number;intent?:{intent:string;target_kind:string;needs_clarification:boolean}}};
export async function command(path:string){
  const response=await fetch('/api/'+path,{method:'POST'});
  const data=await response.json();
  if(!response.ok)throw new Error(data.detail||'Request failed');
  return data;
}
