import json, os, copy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR,'src','EMIL','demand','demand_dictionaries','project_nodal_split','TYNDP_2026_Scenarios_ch4hhp_topology.json')
OUTPUT_FILE = os.path.join(SCRIPT_DIR,'src','EMIL','demand','demand_dictionaries','project_nodal_split','TYNDP_2026_Scenarios_ch4hhp_topology_transformed.json')

CANONICAL = [
    'AT00','BE00','BG00','CY00','CZ00','DE00','DKE1','DKW1','EE00','ES00','FI00','FR00',
    'GR00','GR03','HR00','HU00','IE00',
    'ITCA','ITCN','ITCS','ITN1','ITS1','ITSA','ITSI',
    'LT00','LUG1','LV00','MD00','MK00','MT00','NL00',
    'NOM1','NON1','NOS1','NOS2','NOS3',
    'PL00','PT00','RO00','RS00',
    'SE01','SE02','SE03','SE04',
    'SI00','SK00','UKNI'
]

EXCEPTION = { 'UK00':'UKNI', 'LU00':'LUG1' }
ZONE='Zone 1'

def load():
    with open(INPUT_FILE,'r',encoding='utf-8') as f: return json.load(f)

def save(data):
    with open(OUTPUT_FILE,'w',encoding='utf-8') as f: json.dump(data,f,indent=2)

def transform(data):
    renamed=added=removed=0
    # rename existing
    for country,cdata in data.items():
        if ZONE not in cdata: continue
        z=cdata[ZONE]
        for k in list(z.keys()):
            if k.startswith('B-Methane_Heat-'): continue
            base = EXCEPTION.get(k,k)
            new_key = f'B-Methane_Heat-{base}_HCH4'
            if new_key in z: continue
            z[new_key]=z.pop(k); renamed+=1
    # add missing splits
    for country,cdata in data.items():
        if ZONE not in cdata: continue
        z=cdata[ZONE]
        template_key = next((kk for kk in z.keys() if kk.startswith('B-Methane_Heat-')), None)
        if not template_key: template_key = next(iter(z.keys()), None)
        template_val = z.get(template_key)
        if not template_val: continue
        for base in CANONICAL:
            if base[:2]!=country: continue
            full=f'B-Methane_Heat-{base}_HCH4'
            if full not in z:
                z[full]=copy.deepcopy(template_val); added+=1
    # remove lingering short
    for country,cdata in data.items():
        if ZONE not in cdata: continue
        z=cdata[ZONE]
        for k in list(z.keys()):
            if '-' not in k:
                del z[k]; removed+=1
    return renamed,added,removed

def main():
    data=load()
    r,a,rm = transform(data)
    save(data)
    print(f"Transformed methane topology -> {OUTPUT_FILE}")
    print(f"Renamed={r} Added={a} Removed={rm}")
    # quick verify count
    missing=[]
    for base in CANONICAL:
        country=base[:2]
        key=f'B-Methane_Heat-{base}_HCH4'
        if country not in data or ZONE not in data[country] or key not in data[country][ZONE]:
            missing.append(key)
    if missing:
        print(f"Missing {len(missing)} nodes:")
        for m in missing[:20]: print('  ',m)
    else:
        print("All canonical nodes present.")

if __name__=='__main__':
    main()
