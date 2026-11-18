import csv
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('frontal_audit', Path(__file__).parents[1] / 'scripts/audit_frontal_review.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_review_requires_valid_box_and_reviewer(tmp_path):
    queue = tmp_path / 'queue.csv'
    queue.write_text('source_image_id,width,height,view_review,diastema_review,crowding_review,reviewer\na,100,100,frontal,present,absent,Reviewer\n')
    boxes = tmp_path / 'boxes.csv'
    boxes.write_text('source_image_id,target,x_min,y_min,x_max,y_max,reviewer,review_state\n')
    assert module.audit(queue, boxes)['reviewed_candidates'] == 0
    with boxes.open('a', newline='') as f:
        csv.writer(f).writerow(['a', 'upper_midline_diastema', 10, 10, 110, 40, 'Reviewer', 'approved'])
    result = module.audit(queue, boxes)
    assert result['reviewed_candidates'] == 0
    assert any('geometry' in x for x in result['issues'])
    boxes.write_text('source_image_id,target,x_min,y_min,x_max,y_max,reviewer,review_state\na,upper_midline_diastema,10,10,50,40,Reviewer,approved\n')
    assert module.audit(queue, boxes)['reviewed_candidates'] == 1
