"""Illustrated two-page A4 reference. Diagrams use resolution-independent vectors."""
from pathlib import Path
import math, random, shutil
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/pdf/IBEX_Machine_Learning_Cheat_Sheet.pdf'
OUT.parent.mkdir(parents=True,exist_ok=True)
for name,file in [('Calibri','calibri.ttf'),('Calibri-Bold','calibrib.ttf')]:
    pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
pdfmetrics.registerFontFamily('Calibri',normal='Calibri',bold='Calibri-Bold',italic='Calibri',boldItalic='Calibri-Bold')
W,H=A4; M=30; G=12; CW=(W-2*M-G)/2; TW=(W-2*M-2*G)/3
NAVY='#172B45'; GREEN='#087F73'; BLUE='#4772BB'; ORANGE='#C26720'; GRAY='#526274'; PALE='#EAF5F2'; BORDER='#CFDDD9'
c=canvas.Canvas(str(OUT),pagesize=A4,pageCompression=1)
c.setTitle('IBEX Machine Learning Studio - Illustrated Machine Learning Cheat Sheet')
c.setAuthor('Morteza Khorsand Nikoo')
def color(v): return colors.HexColor(v)
def label(x,y,s,size=9,bold=False,ink=NAVY):
    c.setFillColor(color(ink));c.setFont('Calibri-Bold' if bold else 'Calibri',size);c.drawString(x,y,s)
def para(x,top,s,width,size=9.2,ink=NAVY):
    p=Paragraph(s,ParagraphStyle('p',fontName='Calibri',fontSize=size,leading=size+2.3,textColor=color(ink)))
    _,h=p.wrap(width,1000);p.drawOn(c,x,top-h);return top-h
def line(x1,y1,x2,y2,ink=GREEN,width=1):
    c.setStrokeColor(color(ink));c.setLineWidth(width);c.line(x1,y1,x2,y2)
def arrow(x1,y1,x2,y2,ink=GREEN):
    line(x1,y1,x2,y2,ink,1);a=math.atan2(y2-y1,x2-x1)
    for off in [-.5,.5]: line(x2,y2,x2-5*math.cos(a+off),y2-5*math.sin(a+off),ink,1)
def dot(x,y,ink=GREEN,r=2.2,shape=0):
    c.setFillColor(color(ink));c.setStrokeColor(color(ink))
    if shape: c.rect(x-r,y-r,2*r,2*r,stroke=0,fill=1)
    else: c.circle(x,y,r,stroke=0,fill=1)
def card(x,y,w,h,title,sub=None):
    c.setStrokeColor(color(BORDER));c.setFillColor(colors.white);c.setLineWidth(.6)
    c.roundRect(x,y,w,h,7,stroke=1,fill=1);label(x+11,y+h-20,title,11,True,GREEN)
    if sub: label(x+11,y+h-33,sub,8,False,GRAY)
def axes(x,y,w,h,xlabel=None,ylabel=None):
    line(x,y,x+w,y,GRAY,.55);line(x,y,x,y+h,GRAY,.55)
    if xlabel: label(x+w-42,y-11,xlabel,7.3,ink=GRAY)
    if ylabel: label(x-2,y+h+5,ylabel,7.3,ink=GRAY)
def curve(x,y,w,h,fn,ink=GREEN,dash=None):
    c.saveState();c.setStrokeColor(color(ink));c.setLineWidth(1.5)
    if dash: c.setDash(*dash)
    p=c.beginPath()
    for i in range(81):
        u=i/80;xx=x+w*u;yy=y+h*fn(u)
        if i==0:p.moveTo(xx,yy)
        else:p.lineTo(xx,yy)
    c.drawPath(p);c.restoreState()
def header(page,title,sub):
    label(M,H-27,'IBEX / MACHINE LEARNING STUDIO',10,True,GREEN)
    label(M,H-53,title,23,True);label(M,H-69,sub,9.4,ink=GRAY)
    line(M,H-80,W-M,H-80,GREEN,.9);line(M,42,W-M,42,BORDER,.6)
    label(M,29,'Morteza Khorsand Nikoo | morteza.khorsandnikoo@uibk.ac.at',8,ink=GRAY)
    label(M,17,'A4 | Print at 100% | Illustrative diagrams, not measured results | September 2026',7.4,ink=GRAY)
    label(W-M-22,29,f'{page} / 2',8,ink=GRAY)

header(1,'Machine learning, visually','Choose the task → prepare the data → train → evaluate → predict and save.')
rng=random.Random(71)
for i,(title,sub) in enumerate([('1  Regression','Predict a number'),('2  Classification','Predict a category'),('3  Clustering','Discover groups; no target')]):
    x=M+i*(TW+G);card(x,591,TW,158,title,sub);ax=x+22;ay=636;aw=TW-40;ah=61
    axes(ax,ay,aw,ah)
    if i==0:
        for j in range(15):
            u=(j+1)/17;v=.12+.73*u+rng.uniform(-.12,.12);dot(ax+u*aw,ay+v*ah)
        curve(ax,ay,aw,ah,lambda u:.12+.73*u)
        para(x+11,621,'Start: <b>Linear Regression</b><br/>Curvature? Try polynomial or trees.',TW-22,8.6)
    elif i==1:
        line(ax+aw*.15,ay+ah*.9,ax+aw*.85,ay+ah*.1,GRAY,1)
        for j in range(22):
            u=rng.uniform(.05,.95);v=rng.uniform(.05,.95);cls=int(u+v>1)
            dot(ax+u*aw,ay+v*ah,BLUE if cls else GREEN,2.3,cls)
        para(x+11,621,'Start: <b>Logistic Regression</b><br/>Shapes show two class labels.',TW-22,8.6)
    else:
        for k,(u,v) in enumerate([(.2,.25),(.5,.77),(.8,.28)]):
            for j in range(9): dot(ax+(u+rng.uniform(-.1,.1))*aw,ay+(v+rng.uniform(-.12,.12))*ah,[GREEN,BLUE,ORANGE][k],2.1,k==1)
        para(x+11,621,'Start: <b>K-Means</b> for compact groups.<br/>Density + noise? Try DBSCAN.',TW-22,8.6)
for i,t in enumerate(['Import CSV','Inspect','Prepare','Train','Evaluate','Save']):
    x=M+i*91;c.setFillColor(color(PALE));c.roundRect(x,547,80,29,4,fill=1,stroke=0)
    label(x+9,558,t,9,True)
    if i<5:arrow(x+82,561,x+89,561)

card(M,312,CW,220,'Scale before distance-based learning','Different units can dominate similarity.')
px=M+24;py=418;pw=CW-47
label(px,489,'Raw feature ranges',8.5,True)
line(px,472,px+pw,472,GRAY,.5)
line(px,477,px+pw*.85,477,BLUE,4);label(px+pw*.86,474,'A',8,True,BLUE)
line(px,463,px+pw*.14,463,GREEN,4);label(px+pw*.17,460,'B',8,True,GREEN)
arrow(px+pw*.47,452,px+pw*.47,438)
line(px,py,px+pw*.85,py,BLUE,4);line(px,py-10,px+pw*.85,py-10,GREEN,4)
label(px,py+10,'MinMax: both observed ranges become 0-1',8)
para(M+11,393,'<b>ZScore</b> = (x - mean) / SD<br/><b>MinMax</b> = (x - min) / (max - min)<br/><b>MaxAbs</b> = x / max(|x|)',CW-22,9.2)
para(M+11,346,'Useful for KNN, SVM, neural nets and distance-based clustering. Keep category-code mappings.',CW-22,8.8)
x=M+CW+G;card(x,312,CW,220,'PCA: a new view of the features','Reduce dimensions; retain the original input meaning.')
ax=x+33;ay=414;aw=CW-66;ah=66;axes(ax,ay,aw,ah,'Feature 1','Feature 2')
for j in range(23):
    u=(j+1)/25;dot(ax+u*aw,ay+(.12+.72*u+rng.uniform(-.09,.09))*ah,GREEN,1.7)
arrow(ax+9,ay+7,ax+aw-4,ay+ah-1,ORANGE);label(ax+aw-28,ay+ah+8,'PC1',8,True,ORANGE)
para(x+11,393,'<b>PC1</b> follows the greatest feature variation. Subsequent components capture remaining variation.',CW-22)
para(x+11,359,'Preserve the target with <b>Last column is target (Y)</b>. Predict using original features when the model stores its PCA recipe.',CW-22,8.8)
para(x+11,324,'95% variance retained ≠ 95% prediction accuracy.',CW-22,8.4)

card(M,91,CW,207,'K-fold: every row tested once','Example: 5 folds; shaded block = held-out test rows.')
gx=M+52;gy=242;bw=(CW-76)/5
for r in range(5):
    label(M+12,gy-r*16+3,f'Fold {r+1}',8)
    for k in range(5):
        c.setFillColor(color(GREEN if k==r else '#EDF1F4'));c.setStrokeColor(color(BORDER))
        c.rect(gx+k*bw,gy-r*16,bw-3,12,stroke=1,fill=1)
        if k==r:label(gx+k*bw+4,gy-r*16+3,'Test',6.8,True,'#FFFFFF')
para(M+11,162,'<b>Combined accuracy:</b> total correct held-out predictions / total held-out rows.<br/><b>Combined R²:</b> calculated from pooled predictions; not the mean fold R².',CW-22,8.8)
para(M+11,113,'Training counts repeat rows across folds. Read the Combined <b>test</b> row for held-out results.',CW-22,8.3)
x=M+CW+G;card(x,91,CW,207,'Recognize overfitting','Compare training and validation performance.')
ax=x+31;ay=179;aw=CW-61;ah=66;axes(ax,ay,aw,ah,'Complexity','Error')
curve(ax,ay,aw,ah,lambda u:.7*math.exp(-4*u)+.08,GREEN)
curve(ax,ay,aw,ah,lambda u:.18+2*(u-.5)**2,ORANGE,(3,2))
label(ax+aw-53,ay+9,'Train',7.6,True,GREEN);label(ax+aw-53,ay+ah-5,'Validation',7.6,True,ORANGE)
para(x+11,156,'Train error falls while validation error rises? Reduce complexity, inspect leakage and seek more useful data.',CW-22,8.8)
para(x+11,116,'<b>IBEX:</b> preprocessing is not refitted inside folds. Splits are random, not time-, group- or class-aware. Scores are exploratory.',CW-22,8.3)
c.showPage()

header(2,'Read the plots. Choose the model.','Use several diagnostics together; one impressive score is not enough.')
card(M,564,CW,185,'Regression: actual vs predicted','Points near the diagonal indicate closer predictions.')
ax=M+33;ay=624;aw=119;ah=72;axes(ax,ay,aw,ah,'Actual','Predicted')
line(ax,ay,ax+aw,ay+ah,GRAY,.7)
for j in range(16):
    u=(j+1)/18;dot(ax+u*aw,ay+max(.03,min(.97,u+rng.uniform(-.13,.13)))*ah,GREEN,2)
para(M+166,701,'<b>R² ↑</b><br/>1: perfect<br/>0: mean baseline<br/>&lt; 0: worse',CW-178,8.8)
para(M+11,605,'<b>MAE ↓</b> = mean |actual - predicted|<br/><b>RMSE ↓</b> = √mean(error²); penalizes large errors.<br/>Error units follow the evaluated target scale.',CW-22,8.8)
x=M+CW+G;card(x,564,CW,185,'Residuals: inspect the pattern','Residual = actual - predicted in these illustrations.')
for k in range(2):
    ax=x+24+k*119;ay=635;aw=94;ah=51;axes(ax,ay,aw,ah)
    line(ax,ay+ah/2,ax+aw,ay+ah/2,GRAY,.5)
    for j in range(13):
        u=(j+1)/15;v=.5+rng.uniform(-.24,.24) if k==0 else .22+2.7*(u-.5)**2
        dot(ax+u*aw,ay+v*ah,GREEN if k==0 else ORANGE,1.8)
    label(ax,ay-13,'Random spread' if k==0 else 'Curved pattern',8.2,True)
para(x+11,605,'<b>Left:</b> no obvious structure.<br/><b>Right:</b> the model may miss a nonlinear trend.<br/>A fan-shaped spread suggests changing error variance.',CW-22,8.8)

card(M,351,CW,201,'Classification: confusion matrix','Rows = actual; columns = predicted.')
gx=M+76;gy=411;s=39
label(gx,507,'Predicted',8,True);label(gx+10,493,'Yes',8);label(gx+s+9,493,'No',8)
label(M+13,470,'Actual Yes',8);label(M+13,431,'Actual No',8)
for r,vals in enumerate([['TP  40','FN  10'],['FP   5','TN  45']]):
    for k,v in enumerate(vals):
        c.setFillColor(color(PALE if r==k else '#FBF0E6'));c.setStrokeColor(color('#FFFFFF'))
        c.rect(gx+k*s,gy+(1-r)*s,s,s,fill=1,stroke=1)
        label(gx+k*s+4,gy+(1-r)*s+16,v,8.3,True)
para(M+169,487,'Example:<br/><b>Accuracy</b> 85%<br/><b>Precision</b> 88.9%<br/><b>Recall</b> 80%',CW-181,8.3)
para(M+11,398,'Precision = TP / (TP + FP); recall = TP / (TP + FN).<br/>F1 = 2PR / (P + R). Macro F1 weights classes equally.<br/>Accuracy can hide poor minority-class recall.',CW-22,8.4)
x=M+CW+G;card(x,351,CW,201,'Clustering: groups and noise','Choose features that define meaningful similarity.')
ax=x+25;ay=420;aw=CW-48;ah=73
for k,(u,v) in enumerate([(.23,.55),(.7,.48)]):
    for j in range(17):
        a=j*2.4;rad=.025+(j%5)*.025
        dot(ax+(u+rad*math.cos(a))*aw,ay+(v+rad*2*math.sin(a))*ah,GREEN if k==0 else BLUE,2.1,k==1)
for u,v in [(.04,.1),(.47,.95),(.95,.13)]:
    xx=ax+u*aw;yy=ay+v*ah;line(xx-2,yy-2,xx+2,yy+2,GRAY);line(xx-2,yy+2,xx+2,yy-2,GRAY)
dot(ax+2,ay-6,GREEN,2);label(ax+8,ay-9,'Group A',8,ink=GRAY)
dot(ax+67,ay-6,BLUE,2,1);label(ax+73,ay-9,'Group B',8,ink=GRAY)
line(ax+128,ay-8,ax+132,ay-4,GRAY);line(ax+128,ay-4,ax+132,ay-8,GRAY)
label(ax+138,ay-9,'Noise',8,ink=GRAY)
para(x+11,398,'<b>DBSCAN:</b> can mark noise as -1. K-Means assigns every row to a center. <b>Silhouette:</b> higher usually means better separation; not useful with one cluster.',CW-22,8.7)

label(M,331,'ALGORITHM MAP / start simple, then compare held-out results',11,True,GREEN)
reg=[('Linear Regression','rate, iterations, L1/L2'),('Polynomial','degree; powers, no interactions'),('Normal Equation','ridge stabilization'),('Locally Weighted','tau bandwidth, lambda'),('Random Forest','trees, depth, minLeaf'),('Neural Net (MLP)','layers, size, epochs, rate'),('RBF Surrogate','epsilon width, smoothing'),('FastTree','trees, leaves, rate'),('LightGBM','iterations, leaves, rate'),('Online Gradient Descent','iterations, rate'),('XGBoost','rounds, depth, eta, subsample')]
cls=[('Logistic Regression','linear baseline'),('K-Nearest Neighbors','K; scale matters'),('Naive Bayes','Gaussian baseline'),('SVM (linear)','C; no RBF kernel'),('Random Forest','trees, depth, minLeaf'),('FastTree','boosted one-vs-all trees'),('LightGBM','trainer defaults in IBEX'),('XGBoost','rounds, depth, rate'),('Neural Net (ANN)','layers, size, iterations, rate')]
clust=[('K-Means','K, iterations; nearest center'),('DBSCAN','Eps radius, MinPts density'),('Hierarchical','K; single / average / complete'),('Fuzzy C-Means','K, fuzziness M &gt; 1')]
for i,(title,items) in enumerate([('11 regressors',reg),('9 classifiers',cls),('4 clusterers',clust)]):
    x=M+i*(TW+G);card(x,58,TW,260,title);yy=285
    for name,desc in items: yy=para(x+10,yy,f'<b>{name}</b> · {desc}',TW-20,8.15)-4
    if i==1: para(x+10,yy-2,'Some shared controls are unused by specific classifiers. Check the full guide before tuning.',TW-20,8)
    if i==2:
        yy=para(x+10,yy-8,'<b>Before trusting a result</b><br/>Remove leakage; check class balance and data ranges. Keep a separate final evaluation dataset.',TW-20,8.5)
        para(x+10,yy-12,'<b>In IBEX</b><br/>F1 opens help. Save tables, charts and your model. Reset page does not undo preparation.',TW-20,8.5)
    if yy<67: raise ValueError('Algorithm map overflow')
c.save()
shutil.copyfile(OUT,ROOT/'Help/IBEX_Machine_Learning_Cheat_Sheet.pdf')
print(OUT)
