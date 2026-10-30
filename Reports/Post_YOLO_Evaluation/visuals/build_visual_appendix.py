"""Draw five vector diagram pages from saved benchmark traces; preserve pages 1-6.

Requires reportlab and pypdf. No inference or network access is performed.
Usage: python build_visual_appendix.py --report ../Post_YOLO_Evaluation.pdf
"""
import argparse
import collections
import io
import json
import math
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / 'visualization_data.json').read_text())
W, H = A4
L, WIDTH = 44, 507
INK = '#153448'
TEXT = '#273a46'
MUTED = '#536779'
LINE = '#cbd9e0'
PALE = '#edf3f6'
BLUE = '#2876a3'
GREEN = '#15765d'
GREEN_PALE = '#e8f5ef'
AMBER = '#a66b16'
AMBER_PALE = '#fff4df'
GRAY = '#8b9ba5'


def paragraph(c, text, x, top, width, size=9, color=TEXT, bold=False):
    style = ParagraphStyle('diagram', fontName='Helvetica-Bold' if bold else 'Helvetica',
                           fontSize=size, leading=size * 1.35, textColor=colors.HexColor(color))
    p = Paragraph(text, style)
    _, height = p.wrap(width, H)
    p.drawOn(c, x, H - top - height)
    return height


def line(c, x1, y1, x2, y2, color=LINE, width=1, dashed=False):
    c.saveState()
    c.setStrokeColor(colors.HexColor(color))
    c.setLineWidth(width)
    if dashed:
        c.setDash(3, 3)
    c.line(x1, H-y1, x2, H-y2)
    c.restoreState()


def arrow(c, x1, y1, x2, y2, color=BLUE, width=1.4, dashed=False):
    line(c, x1, y1, x2, y2, color, width, dashed)
    angle = math.atan2(y2-y1, x2-x1)
    for offset in (-0.45, 0.45):
        line(c, x2, y2, x2-5*math.cos(angle+offset), y2-5*math.sin(angle+offset), color, width)


def box(c, x, top, width, height, title, body='', fill=PALE, stroke=LINE, size=9):
    c.setFillColor(colors.HexColor(fill))
    c.setStrokeColor(colors.HexColor(stroke))
    c.setLineWidth(.7)
    c.roundRect(x, H-top-height, width, height, 7, fill=1, stroke=1)
    h = paragraph(c, title, x+10, top+9, width-20, size, bold=True)
    if body:
        bh = paragraph(c, body, x+10, top+13+h, width-20, size-0.5)
        assert 13+h+bh <= height-5, (title, height, h, bh)


def start(c, title, subtitle, page):
    paragraph(c, title, L, 34, WIDTH, 22, INK, True)
    paragraph(c, subtitle, L, 69, WIDTH, 9, MUTED)
    line(c, L, H-35, L+WIDTH, H-35)
    paragraph(c, 'POST-YOLO EVALUATION | Visual explanations from recorded traces', L, H-29, WIDTH-30, 7.5, MUTED)
    paragraph(c, str(page), L+WIDTH-12, H-29, 20, 7.5, MUTED)


def evidence(c, top=96):
    for i, item in enumerate(DATA['evidence']):
        box(c, L+i*173, top, 161, 47, f"{item['id']} | {item['label']} | {item['confidence_percent']:.2f}%",
            'Above 50% review cutoff' if item['confidence'] >= .5 else 'Tentative: verification target',
            fill=PALE if item['confidence'] >= .5 else AMBER_PALE, size=9)


NAMES = {'Breadth-First Search (BFS)': 'BFS - breadth-first search',
         'Depth-First Search (DFS)': 'DFS - depth-first search',
         'Iterative Deepening Search (IDS)': 'IDS - iterative deepening',
         'Uniform-Cost Search (UCS)': 'UCS - uniform-cost search',
         'Greedy Best-First Search (GBFS)': 'Greedy best-first search',
         'A* Search': 'A* search'}
SHORT = {'verify:d2': 'P', 'verify:d3': 'K'}
PATHS = [(), ('P',), ('K',), ('P', 'K'), ('K', 'P')]


def short_path(path):
    return tuple(SHORT[s] for s in path)


def node_values():
    values = {}
    for record in DATA['search'].values():
        for v in record['traversal']:
            key = short_path(v['path'])
            value = (v['g'], v['h'])
            assert key not in values or values[key] == value
            values[key] = value
    assert set(values) == set(PATHS)
    return values


VALUES = node_values()


def tree_panel(c, top, key, description, sequence, priority=None):
    result = DATA['search'][key]
    visits = collections.defaultdict(list)
    for v in result['traversal']:
        visits[short_path(v['path'])].append(v['visit'])
    selected = short_path([s['choice'] for s in result['steps']])
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.HexColor(LINE))
    c.roundRect(L, H-top-183, WIDTH, 183, 7, fill=1, stroke=1)
    paragraph(c, NAMES[key], L+12, top+8, WIDTH-24, 11, INK, True)
    pos = {(): (L+143, top+42), ('P',): (L+68, top+92), ('K',): (L+218, top+92),
           ('P','K'): (L+68, top+142), ('K','P'): (L+218, top+142)}
    for p in PATHS[1:]:
        parent = p[:-1]
        x1,y1 = pos[parent]
        x2,y2 = pos[p]
        active = p == selected[:len(p)]
        arrow(c, x1, y1+30, x2, y2, GREEN if active else LINE, 2 if active else 1)
    for p in PATHS:
        x,y = pos[p]
        active = p == selected[:len(p)]
        visited = p in visits
        c.setFillColor(colors.HexColor(GREEN_PALE if active else PALE if visited else '#fafbfc'))
        c.setStrokeColor(colors.HexColor(GREEN if active else BLUE if visited else LINE))
        c.roundRect(x-46, H-y-30, 92, 30, 5, fill=1, stroke=1)
        label = 'Start' if not p else ' > '.join(p)
        c.setFont('Helvetica-Bold', 8)
        c.setFillColor(colors.HexColor(INK if visited else GRAY))
        c.drawCentredString(x, H-y-11, label)
        g,h = VALUES[p]
        metric = f'g={g:.3f}'
        if priority == 'h': metric = f'h={h:.3f}'
        if priority == 'f': metric = f'f={g+h:.3f}'
        c.setFont('Helvetica', 7.6)
        c.drawCentredString(x, H-y-23, metric)
        if visited:
            badge = ','.join(map(str, visits[p]))
            bw = max(18, 5*len(badge)+8)
            c.setFillColor(colors.HexColor(BLUE))
            c.roundRect(x+37-bw/2, H-y-4, bw, 13, 5, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.setFont('Helvetica-Bold', 7)
            c.drawCentredString(x+37, H-y, badge)
    x = L+288
    paragraph(c, description, x, top+39, 205, 8.5)
    paragraph(c, sequence, x, top+105, 205, 8.1, MUTED)
    paragraph(c, f"<b>{result['expanded_states']} visits | {result['generated_states']} generated</b><br/>Returned cost: <b>{result['cost']:.3f}</b>",
              x, top+151, 205, 8.5, GREEN)


def search_pages(c):
    start(c, 'Search trees: depth and order', 'Recorded example: COCO 43314 (skiing), first measured repetition', 7)
    evidence(c)
    paragraph(c, '<b>P</b> = verify person d2; <b>K</b> = verify skis d3. d1 needs no review. '
              'Each node is a partial review sequence; both leaves cover both targets. Blue badges show actual visit numbers; '
              'green shows the returned path. Gray nodes were not visited before stopping.', L, 152, WIDTH, 8.4)
    tree_panel(c, 198, 'Breadth-First Search (BFS)',
               'A queue visits all shallower states first. Both one-step choices are visited before the first complete plan. '
               'Equal depth does not imply equal cost.',
               '<b>Visit order:</b> Start > P > K > P,K<br/>First recommendation: verify person.')
    tree_panel(c, 389, 'Depth-First Search (DFS)',
               'A stack follows the first branch to a complete plan. It stops before visiting K first, so it misses '
               'the cheaper complete plan in this example.',
               '<b>Visit order:</b> Start > P > P,K<br/>First recommendation: verify person.')
    tree_panel(c, 580, 'Iterative Deepening Search (IDS)',
               'Depth-limited DFS restarts with limits 0, 1 and 2. Multiple numbers on a node show revisits. '
               'It returns the same plan as BFS here.',
               '<b>Limits:</b> 0: Start<br/>1: Start > P > K<br/>2: Start > P > P,K')
    c.showPage()
    start(c, 'Search trees: cost and heuristics', 'Same two targets and possible tree | Different rules choose the next state', 8)
    paragraph(c, '<b>g</b> is cost already paid; <b>h</b> estimates remaining cost; <b>f = g + h</b>. '
              'A review costs 1 unit plus movement between normalized box centers. Starting at image center, '
              'K costs 1.117 and P costs 1.200; moving between them adds a 1.083 review step.', L, 98, WIDTH, 9)
    paragraph(c, 'P > K totals 2.283; K > P totals 2.200. At either one-step node, h = 1.083. '
              'These are review-effort units, not milliseconds or robot travel distance. Blue badges = visits; green = selected path.',
              L, 151, WIDTH, 8.4)
    tree_panel(c, 198, 'Uniform-Cost Search (UCS)',
               'Selects the smallest g. K is cheaper than P, but P is still cheaper than either complete plan, '
               'so UCS visits it before returning K > P.',
               '<b>Visit order:</b> Start > K > P > K,P<br/>First recommendation: verify skis.', 'g')
    tree_panel(c, 389, 'Greedy Best-First Search (GBFS)',
               'Selects the smallest h and ignores g. P and K tie at h=1.083; insertion order favors P. '
               'The next goal has h=0, so it stops.',
               '<b>Visit order:</b> Start > P > P,K<br/>First recommendation: verify person.', 'h')
    tree_panel(c, 580, 'A* Search',
               'Selects the smallest f. K has f=2.200 versus P at 2.283. The K > P goal also has f=2.200, '
               'so A* returns it with fewer visits than UCS.',
               '<b>Visit order:</b> Start > K > K,P<br/>First recommendation: verify skis.', 'f')
    c.showPage()


def bar(c, x, top, width, label, value, color=BLUE):
    paragraph(c, label, x, top, width-45, 8.3)
    paragraph(c, f'{100*value:.1f}%', x+width-43, top, 43, 8.3, color, True)
    c.setFillColor(colors.HexColor(PALE))
    c.roundRect(x, H-top-28, width, 8, 3, fill=1, stroke=0)
    c.setFillColor(colors.HexColor(color))
    c.roundRect(x, H-top-28, width*value, 8, 3, fill=1, stroke=0)


def probability_page(c):
    start(c, 'Probability: two calculations', 'Recorded skiing example | Scene estimation and action selection use separate calculations', 9)
    evidence(c)
    left,right = L,L+265
    arrow(c, L+WIDTH/2, 147, left+121, 176)
    arrow(c, L+WIDTH/2, 147, right+121, 176)
    box(c, left, 178, 242, 86, '1A. Estimate the scene',
        'Count each label once at its highest score: skis 58.15%, person 47.66%. '
        'Combine scene priors and label likelihoods for all 20 hypotheses.')
    box(c, right, 178, 242, 86, '1B. Rate each possible review',
        'Keep all three boxes. Look up P(review useful | label, confidence band) '
        'in the hand-authored policy table.')
    arrow(c, left+121, 265, left+121, 282)
    arrow(c, right+121, 265, right+121, 282)
    box(c, left, 284, 242, 51, '2A. Normalize the scene scores', 'posterior = prior x likelihoods / sum of scores')
    box(c, right, 284, 242, 51, '2B. Select the largest review score', 'max(0.15, 0.80, 0.80) = 0.80')
    posterior = DATA['probability']['scene_inference']['posterior']
    for i,(label,value) in enumerate([('Unclear',posterior['unclear']),('Snow area',posterior['snow_area']),
                                    ('Other 18 scenes, combined',1-posterior['unclear']-posterior['snow_area'])]):
        bar(c,left+9,352+i*45,224,label,value)
    for i,item in enumerate(DATA['probability']['decision_inference']['assessments']):
        action_label = 'presence only' if item['kind'] == 'no_action' else 'verify'
        bar(c,right+9,352+i*45,224,f"{item['detection_id']} {item['label']} / {action_label}",item['review_probability'],GREEN if i==1 else BLUE)
    arrow(c,left+121,488,left+121,508)
    arrow(c,right+121,488,right+121,508)
    box(c,left,510,242,81,'3A. Output: unclear',
        'The leading hypothesis is unclear. No specific scene meets the acceptance gate '
        '(two strong informative cues, at least 55%, and a 15-point lead).',fill=AMBER_PALE)
    box(c,right,510,242,81,'3B. Output: verify person d2',
        '80% exceeds the 50% threshold. d2 and d3 tie; the first maximum, d2, is chosen. '
        'This method selects one review, without ordering a plan.',fill=GREEN_PALE,stroke=GREEN)
    paragraph(c,'A concrete scene calculation',L,614,WIDTH,11,INK,True)
    scene = DATA['probability']['scene_inference']
    joints = {k: scene['priors'][k]*math.prod(scene['likelihood_factors'][k].values()) for k in posterior}
    total = sum(joints.values())
    assert math.isclose(joints['unclear']/total,posterior['unclear'])
    paragraph(c,f"Unclear score = 0.30 x 0.18 x 0.55 = <b>{joints['unclear']:.5f}</b>. "
              f"Divide by the sum over 20 scenes, {total:.5f}, to obtain <b>{100*posterior['unclear']:.2f}%</b>. "
              'The implementation uses log scores for numerical stability. Low-confidence label likelihoods '
              'are weakened; detector confidence itself is not the probability of a scene.',L,637,WIDTH,9)
    paragraph(c,'The scene estimate does not feed into the review-score table. Both sets of probabilities are '
              'hand-authored model values, not measured accuracy. The review reference for this image is No Action; '
              'the tentative YOLO labels cause a verification request.',L,706,WIDTH,8.7)
    c.showPage()


def robot_page(c):
    start(c, 'Robot plans: count, match, select', 'Recorded skiing example | A fixed rule table chooses a named action plan', 10)
    evidence(c)
    box(c,L,165,WIDTH,64,'1. Build counts from eligible boxes',
        '<b>Total 3 | strong 1 | tentative 2 | strong people 0 | vehicles / signals / animals / tools 0.</b><br/>'
        'Only scores at least 50% enter strong category counts. The detected person at 47.66% enters the tentative group.')
    arrow(c,L+166,230,L+166,253)
    paragraph(c,'2. Evaluate all 12 rules in priority order',L,258,WIDTH,11,INK,True)
    labels=['Crowded area','People + vehicles','Signals','Animals','Tools','Uncertain detections',
            'Dense objects','Vehicles','People nearby','Empty area','Sparse objects','Moderate objects']
    top=287
    for i,(label,row) in enumerate(zip(labels,DATA['robot']['evaluations'])):
        y=top+i*25
        match=row['matched']
        c.setFillColor(colors.HexColor(GREEN_PALE if match else PALE if i%2==0 else '#ffffff'))
        c.rect(L,H-y-25,325,25,fill=1,stroke=0)
        paragraph(c,f'{i+1:02d}  {label}',L+8,y+6,238,8.7,GREEN if match else TEXT,match)
        paragraph(c,'MATCH' if match else 'no match',L+257,y+6,66,8.1,GREEN if match else MUTED,match)
    x=L+349
    box(c,x,287,158,86,'3. Choose primary plan',
        'First matched action rule wins priority. All matching action plans and their steps are retained.',size=8.5)
    arrow(c,L+326,top+5*25+12,x-7,top+5*25+12,GREEN)
    line(c,x-7,top+5*25+12,x-7,330,GREEN)
    arrow(c,x-7,330,x,330,GREEN)
    arrow(c,x+79,374,x+79,395,GREEN)
    box(c,x,397,158,87,'Uncertain-Detection',
        '<b>Take Action</b><br/>tentative = 2 >= 1<br/>Targets: d2 and d3',fill=GREEN_PALE,stroke=GREEN,size=9)
    arrow(c,x+79,485,x+79,506,GREEN)
    box(c,x,508,158,79,'4. Recommend the step',
        'Acquire another view before using either tentative label for an action.',fill=GREEN_PALE,stroke=GREEN,size=8.8)
    paragraph(c,'Why this differs from the annotation-based plan',L,614,WIDTH,11,INK,True)
    paragraph(c,'The human annotations establish a person in this scene, so the robot reference selects '
              '<b>People-Presence</b>. The detector reports that person below 50%, so the measured algorithm selects '
              '<b>Uncertain-Detection</b>. Both request Take Action, but their primary plan names differ.',L,638,WIDTH,9)
    paragraph(c,'All 12 rules and 18 conditions were evaluated in the recorded call; this diagram does not imply '
              'that checking stops at rule 6. Idle plans apply only when no action rule matches. '
              'The application displays recommendations and does not execute physical robot commands.',L,707,WIDTH,8.7)
    c.showPage()


def llm_page(c):
    start(c, 'LLM: propose, validate, explain', 'Observed application flow for Qwen3-VL 2B | Internal model reasoning is not recorded', 11)
    box(c,L,101,WIDTH,63,'1. Prepare the model input',
        'Image pixels + YOLO labels, IDs, scores and boxes + allowed action choices. '
        'At most 10 boxes; prompts can be shortened further. The skiing call retained all 3 boxes.')
    x=L; mainw=316; side=L+337; sidew=170
    arrow(c,x+158,165,x+158,184)
    box(c,x,186,mainw,67,'2. Vision-language call: choose',
        'Recorded skiing response:<br/><b>outdoor | snow_area | verify:d2</b>',fill=PALE)
    arrow(c,x+158,254,x+158,274)
    box(c,x,276,mainw,64,'3. Validate the structured choice',
        'Check JSON, allowed environment/setting combinations, and whether the selected action and target are offered.')
    arrow(c,x+158,341,x+158,367,GREEN)
    paragraph(c,'accepted',x+168,347,80,7.7,GREEN)
    box(c,x,369,mainw,76,'4. Text-only call: explain',
        'Receives the validated choice and detection evidence, without the image. '
        'The recorded explanation requested another view for d2. This is an output explanation, not a reasoning trace.')
    arrow(c,x+158,446,x+158,466)
    box(c,x,468,mainw,67,'5. Validate the explanation',
        'Check cited IDs and allowed content. One explanation rewrite is attempted when validation rejects the first answer.')
    arrow(c,x+158,536,x+158,560,GREEN)
    box(c,x,562,mainw,62,'6. Display the accepted recommendation',
        '<b>Take Action: verify person d2.</b><br/>Skiing call: 2 model requests, 9.675 s total.',fill=GREEN_PALE,stroke=GREEN)
    # Failure branch from the choice validator; the recorded street example stops here.
    arrow(c,x+mainw,308,side,308,AMBER)
    box(c,side,276,sidew,105,'Rejected choice: street scene',
        'COCO 555050 returned<br/><b>outdoor + shop + no_action</b>.<br/>'
        'The combination is invalid. The explanation stage was skipped.',fill=AMBER_PALE,stroke=AMBER,size=8.4)
    arrow(c,side+85,382,side+85,559,AMBER)
    box(c,side,562,sidew,62,'Fallback display',
        'No usable decision. The GUI shows No Action; metrics count it as unavailable.',fill=AMBER_PALE,stroke=AMBER,size=8.3)
    # Explanation rejection can either trigger one retry or reach the same fallback.
    arrow(c,x+mainw,498,side+85,498,AMBER,dashed=True)
    paragraph(c,'still rejected<br/>after one retry',side+6,465,151,8,AMBER)
    line(c,x-10,500,x-10,405,BLUE,dashed=True)
    line(c,x,500,x-10,500,BLUE,dashed=True)
    arrow(c,x-10,405,x,405,BLUE,dashed=True)
    paragraph(c,'What this diagram establishes',L,651,WIDTH,11,INK,True)
    paragraph(c,'The green route is the saved skiing result, and the amber route is the saved street-scene rejection. '
              'The dashed retry route is implemented behavior and was not needed for the skiing example. '
              'With no eligible detections, the application bypasses the model and displays No Action.',L,674,WIDTH,9)
    paragraph(c,'The diagrams show prompts, responses and application checks. They do not claim access to the LLM\'s '
              'hidden reasoning. Validation acceptance does not establish that an explanation or action is correct.',L,737,WIDTH,8.4)
    c.showPage()


def build(report, output):
    source = PdfReader(report)
    assert len(source.pages) >= 6
    assert 'Ground truth and reference labels' in source.pages[5].extract_text()
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer,pagesize=A4)
    search_pages(c)
    probability_page(c)
    robot_page(c)
    llm_page(c)
    c.save()
    appendix = PdfReader(buffer)
    assert len(appendix.pages) == 5
    writer = PdfWriter()
    for p in list(source.pages)[:6] + list(appendix.pages):
        writer.add_page(p)
    writer.add_metadata({'/Title':'Post-YOLO Evaluation: Metrics, References and Visual Explanations',
                         '/Author':'Benchmark report'})
    # Assemble bytes before writing so an in-place update is safe and repeatable.
    merged = io.BytesIO()
    writer.write(merged)
    output.write_bytes(merged.getvalue())
    print(f'Built {output} (11 pages; first six report pages preserved)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=HERE.parent/'Post_YOLO_Evaluation.pdf')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    build(args.report,args.output or args.report)
