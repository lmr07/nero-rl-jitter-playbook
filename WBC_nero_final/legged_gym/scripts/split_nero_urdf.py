"""Extract the unchanged Nero subtree; preserve meshes, inertia and attachments."""
from pathlib import Path
import copy
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / 'resources/robots/a2nero/urdf/a2-nero.urdf'
target = source.with_name('nero-fixed.urdf')

def split():
    tree = ET.parse(source, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    root = tree.getroot()
    links = {'nero_base_link'}
    joints = []
    pending = list(root.findall('joint'))
    while pending:
        found = [j for j in pending if j.find('parent').get('link') in links]
        if not found:
            break
        for joint in found:
            links.add(joint.find('child').get('link'))
            joints.append(joint)
            pending.remove(joint)
    out = ET.Element('robot', name='nero_fixed')
    out.append(ET.Comment('Extracted from a2-nero.urdf; root is the Nero mounting flange.'))
    for item in root:
        if (item.tag == 'link' and item.get('name') in links) or item in joints or item.tag == 'material':
            out.append(copy.deepcopy(item))
    ET.ElementTree(out).write(target, encoding='utf-8', xml_declaration=True)
    print(target)

if __name__ == '__main__':
    split()
