"""Restore only the commissioned Nero PEAK interface; never send robot commands."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def discover():
    matches=[]
    for net in Path('/sys/class/net').iterdir():
        dev=(net/'device').resolve()
        hosts=re.findall(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',str(dev))
        driver=(net/'device/driver').resolve().name
        if hosts and hosts[-1]=='0000:00:14.0' and driver=='peak_usb':
            usb=dev.parent
            if (usb/'idVendor').read_text().strip()=='0c72' and (usb/'idProduct').read_text().strip()=='000c':
                matches.append(net.name)
    if len(matches)!=1:
        raise RuntimeError(f'Expected one Nero PEAK 0c72:000c on PCI host 0000:00:14.0, found {matches}. Check cabling; do not guess can0/can1.')
    return matches[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    name=discover()
    state=json.loads(subprocess.check_output(['ip','-j','-details','link','show','dev',name],text=True))[0]
    print(json.dumps(dict(discovered_interface=name,role='Nero',pci_host='0000:00:14.0',
                         usb_id='0c72:000c',driver='peak_usb',link=state),indent=2),flush=True)
    if not args.apply:
        print('Preview only. Run with sudo and --apply to restore can_nero at 1 Mbps.');return
    if name!='can_nero' and Path('/sys/class/net/can_nero').exists():
        raise RuntimeError('can_nero already belongs to another device; refusing to change either')
    info=state['linkinfo']['info_data']
    if 'UP' in state['flags']:
        if name=='can_nero' and info.get('bittiming',{}).get('bitrate')==1000000 and info.get('state')=='ERROR-ACTIVE':
            print('can_nero already ready; no changes');return
        raise RuntimeError('Interface is already UP; stop its controller and inspect before reconfiguration')
    if os.geteuid()!=0:
        raise RuntimeError('Link configuration needs root: sudo python3 scripts/prepare_nero_can.py --apply')
    if name!='can_nero':
        subprocess.run(['ip','link','set','dev',name,'name','can_nero'],check=True)
    subprocess.run(['ip','link','set','dev','can_nero','type','can','bitrate','1000000'],check=True)
    subprocess.run(['ip','link','set','dev','can_nero','txqueuelen','1000'],check=True)
    subprocess.run(['ip','link','set','dev','can_nero','up'],check=True)
    subprocess.run(['ip','-details','link','show','dev','can_nero'],check=True)
    print('Only Nero host link restored. No CAN data/control frames sent; no motor enable.')

if __name__=='__main__':main()
