"""Pure v4 selector over closed raw-event summaries; no legacy dependencies."""
from __future__ import annotations
import itertools, math
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Mapping, Sequence
try:
    from common_bank_successor_v4_schedule import AZIMUTHS,ELEVATIONS,CANDIDATE_IDS,CHARACTERS,view_id
except ModuleNotFoundError:
    from tools.common_bank_successor_v4_schedule import AZIMUTHS,ELEVATIONS,CANDIDATE_IDS,CHARACTERS,view_id

SIDES=("left","right"); STRATA=tuple(itertools.product(CHARACTERS,SIDES))
MARCH_WINDOWS={"left":(75,150),"right":(155,230)}
SIDE_BONES={"left":("left_thigh","left_shank"),"right":("right_thigh","right_shank")}
MIN_CLEAR_FRAMES,MIN_OCCLUDED_FRAMES=5,20
REPEAT_COUNT,SELECT_COUNT,MIN_CARRIERS=3,12,3
EXPECTED_SUBSETS=math.comb(24,12)
SUMMARY_KEYS=frozenset(("found","bone","clear_before_frames","occluded_frames","clear_after_frames"))
LAYOUT=tuple({"view_id":view_id(a,e),"azimuth_deg":a,"elevation_deg":e} for a in AZIMUTHS for e in ELEVATIONS)
class SelectionError(RuntimeError): pass
def _require(ok,message):
    if not ok: raise SelectionError(message)

def side_summary_slacks(summary:Mapping,side:str):
    _require(side in SIDES and isinstance(summary,Mapping),"Invalid side summary.")
    _require(frozenset(summary)==SUMMARY_KEYS,"Closed side-summary schema drifted.")
    _require(isinstance(summary["found"],bool) and summary["bone"] in SIDE_BONES[side],"Event flag or bone drifted.")
    counts=[summary[k] for k in ("clear_before_frames","occluded_frames","clear_after_frames")]
    _require(all(isinstance(v,int) and not isinstance(v,bool) and v>=0 for v in counts),"Event counts must be non-negative integers.")
    lo,hi=MARCH_WINDOWS[side]; _require(sum(counts)<=hi-lo+1,"Event counts exceed registered window.")
    if not summary["found"]: return None
    before,occluded,after=counts
    _require(before>=5 and occluded>=20 and after>=5,"Found event violates contiguous 5/20/5 contract.")
    return min(before,after)-5,occluded-20

def choose_side_summary(bone_summaries:Sequence[Mapping],side:str)->dict:
    _require(not isinstance(bone_summaries,Mapping) and len(bone_summaries)==2,"Exactly thigh and shank required.")
    for value in bone_summaries: side_summary_slacks(value,side)
    _require({v["bone"] for v in bone_summaries}==set(SIDE_BONES[side]),"Side bones drifted.")
    def key(v): return (int(v["found"]),min(v["clear_before_frames"],v["clear_after_frames"]),v["occluded_frames"],v["clear_before_frames"]+v["clear_after_frames"],v["bone"])
    return dict(max(bone_summaries,key=key))

def robust_carrier_profiles(repeats:Mapping)->dict:
    _require(isinstance(repeats,Mapping) and len(repeats)==3,"Exactly three renderer repeats required.")
    per=[]
    for repeat_id in sorted(repeats):
        views=repeats[repeat_id]; _require(set(views)==set(CANDIDATE_IDS),"Repeat candidate set drifted.")
        profile={}
        for view in CANDIDATE_IDS:
            _require(set(views[view])==set(CHARACTERS),"Character strata drifted.")
            for character in CHARACTERS:
                _require(set(views[view][character])==set(SIDES),"Side strata drifted.")
                for side in SIDES: profile[(view,character,side)]=side_summary_slacks(views[view][character][side],side)
        per.append(profile)
    profiles={s:{} for s in STRATA}
    for view in CANDIDATE_IDS:
        for stratum in STRATA:
            values=[p[(view,*stratum)] for p in per]
            if all(v is not None for v in values): profiles[stratum][view]=(min(v[0] for v in values),min(v[1] for v in values))
    return profiles

def _angle(a,b):
    def vector(row):
        az,el=map(math.radians,(row["azimuth_deg"],row["elevation_deg"])); return (-math.cos(el)*math.cos(az),-math.cos(el)*math.sin(az),-math.sin(el))
    x,y=vector(a),vector(b); d=math.sqrt(sum((p-q)**2 for p,q in zip(x,y))); t=math.sqrt(sum((p+q)**2 for p,q in zip(x,y)))
    return int((Decimal(repr(math.degrees(2*math.atan2(d,t))))/Decimal("1e-9")).quantize(Decimal(1),rounding=ROUND_HALF_EVEN))
ANGLE_TABLE={(a["view_id"],b["view_id"]):(0 if a is b else _angle(a,b)) for a in LAYOUT for b in LAYOUT}

def select(repeats:Mapping)->dict:
    profiles=robust_carrier_profiles(repeats); audited=qualifying=0; winner=best=None
    for subset in itertools.combinations(sorted(CANDIDATE_IDS),12):
        audited+=1; chosen=set(subset); counts=[len(chosen&set(profiles[s])) for s in STRATA]
        if min(counts,default=0)<3: continue
        qualifying+=1
        clear=min(sorted((profiles[s][v][0] for v in subset if v in profiles[s]),reverse=True)[2] for s in STRATA)
        occ=min(sorted((profiles[s][v][1] for v in subset if v in profiles[s]),reverse=True)[2] for s in STRATA)
        prefix=(clear,occ,min(counts),sum(counts))
        if best is not None and prefix<best[:4]: continue
        angle=min(ANGLE_TABLE[a,b] for a,b in itertools.combinations(subset,2)); score=(*prefix,angle)
        if best is None or score>best or (score==best and subset<tuple(winner)): best,winner=score,list(subset)
    _require(audited==EXPECTED_SUBSETS,"Exhaustive audit truncated.")
    return {"record":"fs_cts5_common_bank_successor_v4_selection_v1","pass":winner is not None,"failure":None if winner else "zero_qualifying_subsets_terminal_stop","subsets_audited":audited,"subsets_expected":EXPECTED_SUBSETS,"qualifying_subsets":qualifying,"selected_transition_view_ids":winner,"score":list(best) if best else None,"score_order":["worst_stratum_third_best_clear_flank_slack","worst_stratum_third_best_occluded_run_slack","minimum_robust_carrier_count","total_robust_carrier_incidence","minimum_pairwise_angle_nanodeg","lexicographic_view_ids"],"robust_carriers_by_stratum":{f"{c}|{s}":sorted(profiles[(c,s)]) for c,s in STRATA},"probe_rows_reusable_downstream":False,"fresh_downstream_view_count":18 if winner else None}
