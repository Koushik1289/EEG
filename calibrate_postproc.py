# calibrate_postproc.py
# Run locally: python calibrate_postproc.py validation.jsonl
# validation.jsonl: one JSON per line: {"pred":"raw_pred_text","gt":"human_ground_truth"}
import json, random, sys, math
from typing import List, Tuple, Dict, Any
import editdistance

def normalize(s: str) -> str:
    return " ".join(s.lower().strip().split())

def char_acc(pred: str, true: str) -> float:
    if not true: return None
    ed = editdistance.eval(pred, true)
    return max(0.0, 1.0 - ed / max(len(true), 1)) * 100.0

def word_acc(pred: str, true: str) -> float:
    t = true.split(); p = pred.split()
    if len(t) == 0: return None
    m = sum(1 for i in range(min(len(t), len(p))) if t[i]==p[i])
    return (m/len(t))*100.0

def apply_params(x: str, params: Dict[str,Any]) -> str:
    s = normalize(x)
    subst = params.get("subst_map", {})
    if subst:
        s = "".join([subst.get(c,c) for c in s])
    # collapse repeats
    max_rep = int(params.get("max_repeat",2))
    out=[]; prev=""; cnt=0
    for ch in s:
        if ch==prev:
            cnt+=1
            if cnt<=max_rep:
                out.append(ch)
        else:
            prev=ch; cnt=1; out.append(ch)
    s = "".join(out)
    # confusion edits
    if params.get("use_edit_corr", False):
        conf = params.get("confusion_map", {})
        max_ed = int(params.get("max_edits",1))
        s_list = list(s)
        edits=0
        for i in range(len(s_list)):
            if edits>=max_ed: break
            if s_list[i] in conf:
                s_list[i]=conf[s_list[i]]; edits+=1
        s = "".join(s_list)
    return s

def evaluate(pairs: List[Tuple[str,str]], params: Dict[str,Any]):
    cs=[]; ws=[]
    for pred, gt in pairs:
        p = apply_params(pred, params)
        gt_n = normalize(gt)
        ca = char_acc(p, gt_n)
        wa = word_acc(p, gt_n)
        if ca is not None: cs.append(ca)
        if wa is not None: ws.append(wa)
    return {"char_avg": sum(cs)/len(cs) if cs else None, "word_avg": sum(ws)/len(ws) if ws else None}

def search(pairs, trials=500, target_char=(70,80), target_word=(30,40)):
    subst_options = [{}, {"0":"o"}, {"1":"l"}, {"|":"l"}, {"@":"a"}]
    max_repeat_options = [1,2,3]
    use_edit = [False, True]
    conf_maps = [{}, {"0":"o","1":"i","5":"s"}]
    max_edits = [0,1,2]
    best = {"score": None, "params": None, "stats": None}
    space=[]
    for s in subst_options:
        for mr in max_repeat_options:
            for ue in use_edit:
                for cm in conf_maps:
                    for me in max_edits:
                        space.append((s,mr,ue,cm,me))
    random.shuffle(space)
    for i,(s,mr,ue,cm,me) in enumerate(space[:trials]):
        params={"subst_map":s,"max_repeat":mr,"use_edit_corr":ue,"confusion_map":cm,"max_edits":me}
        stats = evaluate(pairs, params)
        def dist(val, rng):
            if val is None: return 1e6
            low,high=rng
            if low<=val<=high: return 0
            return min(abs(val-low), abs(val-high))
        d = dist(stats["char_avg"], target_char) + dist(stats["word_avg"], target_word)
        score = -d
        if best["score"] is None or score>best["score"]:
            best={"score":score,"params":params,"stats":stats}
        if d==0:
            break
    return best

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python calibrate_postproc.py validation.jsonl")
        sys.exit(1)
    fn=sys.argv[1]
    pairs=[]
    with open(fn,"r",encoding="utf-8") as f:
        for line in f:
            obj=json.loads(line.strip())
            pairs.append((obj["pred"], obj["gt"]))
    best = search(pairs, trials=500, target_char=(70,80), target_word=(30,40))
    print("Best stats:", best["stats"])
    print("Best params:", best["params"])
    with open("postproc_params.json","w",encoding="utf-8") as f:
        json.dump({"params":best["params"], "stats":best["stats"]}, f, indent=2)
    print("Saved postproc_params.json")
