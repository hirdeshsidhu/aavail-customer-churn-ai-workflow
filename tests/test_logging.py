from app.logging_utils import log_prediction,read_logs
def test_logging_isolated(tmp_path):
    p=tmp_path/"test.jsonl"; log_prediction(p,{"country":"Singapore"},1,.72)
    rows=read_logs(p); assert len(rows)==1; assert rows[0]["prediction"]==1
