"""Check SDK progress on monotonic time, corroborated by fresh physical CAN status."""
import math

class DriverClock:
 def __init__(self):
  self.stamps=None;self.changed=None
 def check(self,enabled,raw,now):
  if enabled is None or len(enabled[0])!=7 or len(enabled[1])!=7:
   raise RuntimeError('Missing SDK driver status')
  bits,stamps=enabled
  if not all(bits) or not all(math.isfinite(s) for s in stamps):
   raise RuntimeError('SDK driver disabled or invalid timestamp')
  payloads=raw['driver_payloads'];rx=raw['driver_stamps_s']
  if len(payloads)!=7 or len(rx)!=7:
   raise RuntimeError('Incomplete physical driver status')
  for payload,stamp in zip(payloads,rx):
   if payload is None or not math.isfinite(stamp) or not (-.01<=now-stamp<=.2):
    raise RuntimeError('Physical driver feedback stale')
   data=bytes.fromhex(payload)
   if len(data)!=8 or data[5]!=0x40:
    raise RuntimeError('Physical driver disabled or reports fault')
  # An SDK stamp identifies a newly consumed frame, not an elapsed-time clock.
  # Wall-clock steps can move it backwards; monotonic receive progress must persist.
  if self.stamps is None:
   self.stamps=list(stamps);self.changed=[now]*7
  for j,s in enumerate(stamps):
   if s!=self.stamps[j]:self.stamps[j]=s;self.changed[j]=now
   if now-self.changed[j]>.2:raise RuntimeError('SDK driver feedback stopped advancing')
  return [now-t for t in self.changed]
