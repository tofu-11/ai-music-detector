const root=document.documentElement,isDark=()=>root.dataset.theme?root.dataset.theme==="dark":matchMedia("(prefers-color-scheme:dark)").matches;
const SR=11025,$=s=>document.querySelector(s),sleep=ms=>new Promise(r=>setTimeout(r,ms)),css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function rng(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
/* synthetic training clips (stand-in for the real dataset) */
function gen(seed,ai){const r=rng(seed),n=SR*3,o=new Float32Array(n),base=196*Math.pow(2,Math.floor(r()*8)/12),vib=ai?r()*.004:r()*.012+.002,noise=ai?r()*.01:.006+r()*.04,jit=ai?0:r()*.08,pent=[0,2,4,7,9];let t=0,ph=0;
while(t<3){const f=base*Math.pow(2,pent[Math.floor(r()*5)]/12+Math.floor(r()*2)),len=.25+(r()-.5)*jit*2,amp=ai?.6:.4+r()*.4,s=Math.floor(t*SR),e=Math.min(n,Math.floor((t+len)*SR));
for(let i=s;i<e;i++){const k=(i-s)/(e-s),env=Math.min(1,k*20)*Math.min(1,(1-k)*8);ph+=2*Math.PI*f*(1+vib*Math.sin(2*Math.PI*5.5*i/SR))/SR;o[i]+=amp*env*(Math.sin(ph)+.5*Math.sin(2*ph)+.25*Math.sin(3*ph))*.4+(r()-.5)*noise}t+=len}return o}
/* MFCC */
const NF=512,HOP=384,NM=26,NC=13;
function fft(re,im){const n=re.length;for(let i=1,j=0;i<n;i++){let b=n>>1;for(;j&b;b>>=1)j^=b;j^=b;if(i<j){[re[i],re[j]]=[re[j],re[i]];[im[i],im[j]]=[im[j],im[i]]}}
for(let l=2;l<=n;l<<=1){const a=-2*Math.PI/l;for(let i=0;i<n;i+=l)for(let k=0;k<l/2;k++){const c=Math.cos(a*k),s=Math.sin(a*k),u=i+k,v=u+l/2,xr=re[v]*c-im[v]*s,xi=re[v]*s+im[v]*c;re[v]=re[u]-xr;im[v]=im[u]-xi;re[u]+=xr;im[u]+=xi}}}
const mel=f=>2595*Math.log10(1+f/700),imel=m=>700*(Math.pow(10,m/2595)-1);
const FB=(()=>{const p=[],lo=mel(0),hi=mel(SR/2);for(let i=0;i<NM+2;i++)p.push(Math.floor((NF+1)*imel(lo+(hi-lo)*i/(NM+1))/SR));
return Array.from({length:NM},(_,m)=>{const w=new Float32Array(NF/2+1);for(let k=p[m];k<p[m+1];k++)w[k]=(k-p[m])/(p[m+1]-p[m]||1);for(let k=p[m+1];k<p[m+2];k++)w[k]=(p[m+2]-k)/(p[m+2]-p[m+1]||1);return w})})();
function mfcc(x){const F=[],win=Array.from({length:NF},(_,i)=>.5-.5*Math.cos(2*Math.PI*i/(NF-1)));
for(let s=0;s+NF<=x.length;s+=HOP){const re=new Float64Array(NF),im=new Float64Array(NF);for(let i=0;i<NF;i++)re[i]=x[s+i]*win[i];fft(re,im);
const le=FB.map(w=>{let e=0;for(let k=0;k<=NF/2;k++)e+=w[k]*(re[k]*re[k]+im[k]*im[k]);return Math.log(e+1e-10)});
F.push(Array.from({length:NC},(_,c)=>le.reduce((a,v,m)=>a+v*Math.cos(Math.PI*c*(m+.5)/NM),0)))}return F}
function feats(F){const T=F.length,f=[];for(let c=1;c<=12;c++){const m=F.reduce((a,r)=>a+r[c],0)/T;f.push(Math.sqrt(F.reduce((a,r)=>a+(r[c]-m)**2,0)/T))}
let d=0;for(let t=1;t<T;t++)for(let c=1;c<=4;c++)d+=Math.abs(F[t][c]-F[t-1][c]);f.push(d/((T-1)*4));f.push(F.reduce((a,r)=>a+r[1],0)/T);f.push(F.reduce((a,r)=>a+r[2],0)/T);return f}
/* Random Forest (31 stumps, trained here on synthetic clips) */
const NT=31,forest=[];
(function(){const X=[],Y=[];for(let i=0;i<60;i++){const ai=i%2;X.push(feats(mfcc(gen(1000+i,ai))));Y.push(ai)}const r=rng(7);
for(let t=0;t<NT;t++){const idx=Array.from({length:60},()=>Math.floor(r()*60)),fs=[0,0,0,0].map(()=>Math.floor(r()*X[0].length));let b={a:-1};
for(const f of fs)for(const i of idx){const th=X[i][f];for(const dir of[1,-1]){const a=idx.filter(j=>((X[j][f]>th?1:0)^(dir<0?1:0))===Y[j]).length;if(a>b.a)b={a,f,th,dir}}}forest.push(b)}})();
function predict(x){const F=mfcc(x),f=feats(F);let v=0;for(const s of forest)if(((f[s.f]>s.th?1:0)^(s.dir<0?1:0))===1)v++;return{F,p:v/NT}}
/* helpers */
const fmt=s=>Math.floor(s/60)+":"+String(Math.round(s%60)).padStart(2,"0");
function toWav(x){const b=new ArrayBuffer(44+x.length*2),v=new DataView(b),w=(o,s)=>[...s].forEach((c,i)=>v.setUint8(o+i,c.charCodeAt(0)));w(0,"RIFF");v.setUint32(4,36+x.length*2,true);w(8,"WAVEfmt ");v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,SR,true);v.setUint32(28,SR*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);w(36,"data");v.setUint32(40,x.length*2,true);x.forEach((s,i)=>v.setInt16(44+i*2,Math.max(-1,Math.min(1,s))*32767,true));return b}
const V=["upload","result","how","about"];function show(v){V.forEach(k=>$("#v-"+k).classList.toggle("hide",k!==v));document.querySelectorAll("nav [data-nav]").forEach(a=>a.classList.toggle("active",a.dataset.nav===v));$("#hero").classList.toggle("hide",v!=="upload");document.body.dataset.view=v;scrollTo(0,0)}
/* state */
let AC,cur=null,node=null,lastF=null;  // cur={name,size,dur,x(11025 mono),buf}
function showErr(m){const e=$("#err");e.textContent=m||"";e.classList.toggle("hide",!m)}
function setFile(f){cur=f;showErr();$("#fileRow").classList.toggle("hide",!f);$("#drop").classList.toggle("hide",!!f);$("#analyze").disabled=!f;if(f){$("#fName").textContent=f.name;$("#fDur").textContent=fmt(f.dur);$("#fSize").textContent="· "+(f.size/1048576).toFixed(1)+" MB"}}
async function load(file){showErr();const ok=/\.(mp3|wav)$/i.test(file.name);if(!ok)return showErr("Please choose an MP3 or WAV file.");if(file.size>30*1048576)return showErr("File is larger than 30 MB.");
try{AC=AC||new(window.AudioContext||window.webkitAudioContext)();const b=await AC.decodeAudioData(await file.arrayBuffer());if(b.duration>600.5)return showErr("Clip is longer than 10 minutes.");
const ch=b.getChannelData(0),r=b.sampleRate/SR,n=Math.floor(ch.length/r),x=new Float32Array(n);for(let i=0;i<n;i++){let s=0,c=0;for(let k=Math.floor(i*r);k<Math.floor((i+1)*r)&&k<ch.length;k++){s+=ch[k];c++}x[i]=c?s/c:0}
setFile({name:file.name,size:file.size,dur:b.duration,x})}catch(e){showErr("That file could not be read as audio.")}}
function demo(ai){const x=gen(ai?555:777,ai);setFile({name:ai?"demo_clip_a.wav":"demo_clip_b.wav",size:x.length*2+44,dur:x.length/SR,x})}
/* drawing */
function wave(x){const c=$("#wave"),w=c.width=c.clientWidth*2,h=c.height=80,g=c.getContext("2d");g.clearRect(0,0,w,h);g.fillStyle=css("--ink");const bars=Math.floor(w/8);
for(let i=0;i<bars;i++){const s=Math.floor(i/bars*x.length);let m=0;for(let k=0;k<400&&s+k<x.length;k+=8)m=Math.max(m,Math.abs(x[s+k]));const bh=Math.max(4,m*h*1.6);g.fillRect(i*8,(h-bh)/2,3,bh)}}
function heat(F){const c=$("#mfcc"),w=c.width=c.clientWidth*2,h=c.height=140,g=c.getContext("2d"),T=F.length,dark=isDark();
for(let k=0;k<NC;k++){const col=F.map(r=>r[k]),mn=Math.min(...col),mx=Math.max(...col);col.forEach((v,t)=>{const u=(v-mn)/(mx-mn||1);g.fillStyle=`hsl(${75-u*55},${dark?55+u*35:80+u*15}%,${dark?12+u*50:93-u*50}%)`;g.fillRect(t*w/T,k*h/NC,w/T+1,h/NC+1)})}}
function stop(){if(node){try{node.stop()}catch(e){}node=null}$("#play").textContent="▶"}
/* actions */
$("#analyze").onclick=async()=>{const b=$("#analyze");b.disabled=true;b.querySelector("span").textContent="Analyzing…";await sleep(700);const r=predict(cur.x),ai=r.p>=.5,top=Math.round(Math.max(r.p,1-r.p)*100);
$("#rName").textContent=cur.name;$("#rDur").textContent=fmt(cur.dur);$("#verdict").textContent=ai?"AI-generated":"Human-composed";$("#verdict").dataset.kind=ai?"ai":"human";$("#vIcon").textContent=ai?"AI":"♪";$("#meter").dataset.kind=ai?"ai":"human";$("#conf").textContent=top+"%";$("#meter").style.width=top+"%";
$("#other").textContent=(ai?"Human-composed: ":"AI-generated: ")+(100-top)+"%";lastF=r.F;show("result");wave(cur.x);heat(lastF);b.querySelector("span").textContent="Analyze clip";b.disabled=false};
$("#play").onclick=()=>{if(node)return stop();AC=AC||new(window.AudioContext||window.webkitAudioContext)();const b=AC.createBuffer(1,cur.x.length,SR);b.copyToChannel(cur.x,0);node=AC.createBufferSource();node.buffer=b;node.connect(AC.destination);node.onended=()=>{node=null;$("#play").textContent="▶"};node.start();$("#play").textContent="❚❚"};
const toUpload=()=>{stop();setFile(null);show("upload")};
$("#another").onclick=toUpload;$("#back").onclick=()=>{stop();show("upload")};$("#home").onclick=()=>{stop();show("upload")};
$("#remove").onclick=()=>setFile(null);$("#choose").onclick=()=>$("#file").click();$("#file").onchange=e=>{e.target.files[0]&&load(e.target.files[0]);e.target.value=""};
const d=$("#drop");["dragenter","dragover"].forEach(t=>d.addEventListener(t,e=>{e.preventDefault();d.classList.add("over")}));["dragleave","drop"].forEach(t=>d.addEventListener(t,e=>{e.preventDefault();d.classList.remove("over")}));
d.addEventListener("drop",e=>{const f=e.dataTransfer.files[0];f&&load(f)});
$("#demoAI").onclick=()=>demo(1);$("#demoHu").onclick=()=>demo(0);
document.querySelectorAll("[data-goto]").forEach(b=>b.onclick=()=>{stop();show(b.dataset.goto)});$("#nHow").onclick=()=>{stop();show("how")};$("#nAbout").onclick=()=>{stop();show("about")};document.querySelectorAll("[data-go]").forEach(b=>b.onclick=()=>show("upload"));
/* theme */
function applyTheme(){const d=isDark(),b=$("#theme");b.textContent=d?"☀ Light":"☾ Dark";b.setAttribute("aria-label",d?"Switch to light mode":"Switch to dark mode");
if(cur&&lastF&&!$("#v-result").classList.contains("hide")){wave(cur.x);heat(lastF)}}
try{const t=localStorage.getItem("theme");if(t==="light"||t==="dark")root.dataset.theme=t}catch(e){}
$("#theme").onclick=()=>{const t=isDark()?"light":"dark";root.dataset.theme=t;try{localStorage.setItem("theme",t)}catch(e){}applyTheme()};
matchMedia("(prefers-color-scheme:dark)").addEventListener("change",applyTheme);
applyTheme();
