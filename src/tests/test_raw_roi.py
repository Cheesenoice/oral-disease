"""Raw-source derivation must preserve annotation identities and reject tampering."""
import json
import pandas as pd
import pytest
from PIL import Image
from dentalselfie.acquire import digest
from dentalselfie.raw_roi import prepare_raw_rois, retained_annotations


def setup_tree(root):
    for p in ['data/manifests','data/processed/gum_roi_v1','src/configs','raw']:
        (root/p).mkdir(parents=True)
    Image.new('RGB',(200,100),'red').save(root/'raw/source.jpg')
    (root/'raw/source.txt').write_text('0 0.5 0.5 1 1\n3 0.5 0.5 0.2 0.4\n')
    (root/'raw/prepared.txt').write_text('1 0.5 0.5 0.2 0.4\n')
    parent=dict(sample_id='parent',split='train',group_id='group',source_image='raw/source.jpg',
                source_label='raw/source.txt',source_sha256=digest(root/'raw/source.jpg'),
                source_label_sha256=digest(root/'raw/source.txt'),source_width=200,source_height=100,
                label='raw/prepared.txt')
    splits=root/'data/manifests/gum.splits.csv'
    pd.DataFrame([parent,dict(sample_id='no_test_file',split='test')]).to_csv(splits,index=False)
    (splits.parent/'gum.lock.json').write_text(json.dumps({'splits_sha256':digest(splits)}))
    old=root/'data/processed/gum_roi_v1/manifest.csv'
    pd.DataFrame([dict(crop_id='parent_00',parent_id='parent',group_id='group',split='train',
        class_id=1,class_name='mild_inflammation',box_index=0,box_width=4,box_height=4,
        local_sha256='old')]).to_csv(old,index=False)
    (old.parent/'receipt.json').write_text(json.dumps({'manifest_sha256':digest(old)}))
    (root/'src/configs/label_mappings.json').write_text(json.dumps({'gum':{'classes':{'0':None,'3':1}}}))
    (root/'src/configs/roi.json').write_text(json.dumps({'size':224,'padding_rgb':[128]*3}))


def test_original_coordinate_crop_and_unopened_test(tmp_path):
    setup_tree(tmp_path)
    frame,receipt=prepare_raw_rois(tmp_path,workers=1)
    assert receipt['test_images_opened']==0 and len(frame)==1
    assert frame.iloc[0].source_annotation_line==1
    assert frame.iloc[0].raw_box_width==40
    again,_=prepare_raw_rois(tmp_path,workers=1)
    assert again.local_sha256.tolist()==frame.local_sha256.tolist()
    path=tmp_path/'data/processed/gum_raw_roi_v1'/frame.iloc[0].local_path
    path.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='checksum'):
        prepare_raw_rois(tmp_path,workers=1)


def test_source_tampering_is_rejected(tmp_path):
    setup_tree(tmp_path)
    (tmp_path/'raw/source.txt').write_text('3 0.5 0.5 0.9 0.9')
    with pytest.raises(ValueError,match='fingerprint'):
        prepare_raw_rois(tmp_path,workers=1)


def test_mapping_never_guesses_unknown_class():
    with pytest.raises(ValueError,match='Unknown'):
        retained_annotations('9 0.5 0.5 0.1 0.1',{'0':None})
