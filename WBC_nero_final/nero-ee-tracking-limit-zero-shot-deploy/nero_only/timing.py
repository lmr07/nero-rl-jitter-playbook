"""Nonblocking, timestamped trace records. CAN stamps are host software stamps."""
import json
import queue
import threading
import time


class Trace:
    def __init__(self,path):
        self.path=str(path)
        self.queue=queue.Queue(maxsize=20000)
        self.error=None
        self.dropped=0
        self.thread=threading.Thread(target=self._write,name='nero-trace-writer',daemon=True)
        self.thread.start()
        self.emit('trace_header',clock='CLOCK_MONOTONIC',timestamp_source='SocketCAN software timestamp/local echo; NOT device sample or drive-apply timestamp',wall_time=time.time())

    def emit(self,kind,**values):
        record={'kind':kind,'host_monotonic':time.monotonic(),**values}
        try:self.queue.put_nowait(record)
        except queue.Full:self.dropped+=1

    def can_frame(self,msg,stamp,dequeue):
        if not (0x15a<=msg.arbitration_id<=0x160 or 0x251<=msg.arbitration_id<=0x257 or 0x261<=msg.arbitration_id<=0x267 or msg.arbitration_id==0x2a1):return
        self.emit('can_frame',can_id=int(msg.arbitration_id),data=bytes(msg.data).hex(),
            kernel_wall_s=float(msg.timestamp),kernel_monotonic_s=stamp,dequeue_monotonic_s=dequeue,
            reader_queue_s=max(0.,dequeue-stamp),is_rx=bool(msg.is_rx))

    def check(self):
        if self.error is not None or self.dropped:
            raise RuntimeError(f'Timing log incomplete: dropped={self.dropped}, error={self.error!r}')

    def _write(self):
        try:
            with open(self.path,'w',buffering=1) as stream:
                while True:
                    record=self.queue.get()
                    if record is None:break
                    stream.write(json.dumps(record,allow_nan=True)+'\n')
        except BaseException as exc:self.error=exc

    def close(self):
        try:self.queue.put(None,timeout=2)
        except queue.Full:raise RuntimeError('Timing writer queue did not drain')
        self.thread.join(timeout=5)
        if self.thread.is_alive():raise RuntimeError('Timing log did not finish flushing')
        self.check()
