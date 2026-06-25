// Replicate the fixed dynamicMarket pricing for the user's scenario
const TEAMS=12, BUDGET=200, ROSTER=14;
// build pool: 12 QBs at proj 100 (+1 QB at 0), 1 RB at 100 + 40 RBs at 0, 24 WR at 0, 12 TE at 0
let pool=[];
for(let i=0;i<12;i++) pool.push({pos:'QB',proj:100,id:'QB'+i});
pool.push({pos:'QB',proj:0,id:'QB12'});
pool.push({pos:'RB',proj:100,id:'RB_elite'});
for(let i=0;i<40;i++) pool.push({pos:'RB',proj:0,id:'RB'+i});
for(let i=0;i<24;i++) pool.push({pos:'WR',proj:0,id:'WR'+i});
for(let i=0;i<12;i++) pool.push({pos:'TE',proj:0,id:'TE'+i});

const open={QB:12,RB:24,WR:24,TE:12,FLEX:12};
const flexAdd=(pos)=>({RB:.45,WR:.45,TE:.10})[pos]*open.FLEX||0;
const money=TEAMS*BUDGET, spots=TEAMS*ROSTER, disc=Math.max(money-spots,0);
const repl={}, within=new Set();
for(const pos of ['QB','RB','WR','TE']){
  const n=Math.max(0,Math.round(open[pos]+flexAdd(pos)));
  const arr=pool.filter(p=>p.pos===pos).sort((a,b)=>b.proj-a.proj);
  const mi=Math.min(Math.max(n-1,0),arr.length-1);
  repl[pos]=arr.length?(n>0?arr[mi].proj:arr[0].proj):0;       // FIX: marginal startable (rank n)
  arr.slice(0,n).forEach(p=>within.add(p.id));
}
let sumV=0; pool.forEach(p=>{if(within.has(p.id))sumV+=Math.max(p.proj-repl[p.pos],0);});
const per=sumV>0?disc/sumV:0;
function price(p){const v=Math.max(p.proj-repl[p.pos],0); return within.has(p.id)?Math.max(1,Math.round(1+v*per)):(v>0?1:0);}
// OLD (broken) baseline = first bench (arr[n]) for comparison
const replOld={};
for(const pos of ['QB','RB','WR','TE']){const n=Math.max(0,Math.round(open[pos]+flexAdd(pos)));const arr=pool.filter(p=>p.pos===pos).sort((a,b)=>b.proj-a.proj);replOld[pos]=n<arr.length?arr[n].proj:0;}
let sumVo=0; pool.forEach(p=>{if(within.has(p.id))sumVo+=Math.max(p.proj-replOld[p.pos],0);});
const perO=sumVo>0?disc/sumVo:0;
const priceOld=(p)=>{const v=Math.max(p.proj-replOld[p.pos],0);return within.has(p.id)?Math.max(1,Math.round(1+v*perO)):(v>0?1:0);};

const qb=pool.find(p=>p.id==='QB0'), rb=pool.find(p=>p.id==='RB_elite');
console.log('repl baseline (fixed):', repl);
console.log('--- OLD naive VORP engine (baseline=first bench) ---');
console.log('  a 100-VORP QB  -> $'+priceOld(qb)+'   the scarce RB -> $'+priceOld(rb));
console.log('--- FIXED VONA engine (baseline=marginal startable) ---');
console.log('  a 100-VORP QB  -> $'+price(qb)+'   the scarce RB -> $'+price(rb));
