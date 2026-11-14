/* Browser implementation of the paper's engineering input and frozen inverse models. */
(function (global) {
  'use strict';
  const data = global.MagTouchStudy;
  const {heights, betas, weights, templates} = data;
  const grid = new Map(data.grid.map(([region, phi, level, field]) => [region + ':' + phi + ':' + level, field]));
  const mod = (x, n) => ((x % n) + n) % n;
  const distance = (a, b) => Math.hypot(...a.map((x, i) => x - b[i]));

  function project([x, y, z]) {
    const phi = Math.atan2(y, x);
    const side = [20 * Math.cos(phi), 20 * Math.sin(phi), Math.max(0, Math.min(100, z))];
    const dz = Math.max(0, z - 100);
    const norm = Math.hypot(x, y, dz);
    if (norm === 0) return side;
    const cap = [20*x/norm, 20*y/norm, 100+20*dz/norm];
    return distance([x,y,z], side) <= distance([x,y,z], cap) ? side : cap;
  }

  function forward(b, model) {
    let h = b.map((value, i) => (value - model.center[i]) / model.spread[i]);
    for (let i=0; model['arr_'+i]; i+=2) {
      const w=model['arr_'+i], bias=model['arr_'+(i+1)];
      h=bias.map((value, j) => {
        for (let k=0; k<h.length; k++) value += w[k][j]*h[k];
        return model['arr_'+(i+2)] ? Math.tanh(value) : value;
      });
    }
    return project([20*h[0],20*h[1],60*h[2]+60]);
  }

  function nearest(field) {
    const n=Math.hypot(...field);
    if (!(n>0)) throw new Error('Zero field');
    let best=-Infinity, index=0;
    templates.vectors.forEach((vector, i) => {
      let score=0;
      for (let j=0; j<field.length; j++) score += vector[j]*(field[j]/n);
      if (score>best) {best=score; index=i;}
    });
    return templates.xyz[index];
  }

  function fieldAt(region, phi, level) {
    const levels=region==='side' ? heights : betas;
    if (level < levels[0]-1e-7 || level > levels.at(-1)+1e-7) return null;
    const x=Math.max(levels[0],Math.min(levels.at(-1),level));
    const hi=levels.findIndex(v=>v>x);
    const j=hi===-1 ? levels.length-2 : hi-1;
    const vertical=(x-levels[j])/(levels[j+1]-levels[j]);
    const start=45*Math.floor(phi/45), horizontal=(phi-start)/45;
    const stencil=new Map();
    for (const [az, u] of [[start,1-horizontal],[mod(start+45,360),horizontal]]) {
      for (const [h, v] of [[levels[j],1-vertical],[levels[j+1],vertical]]) {
        if (u*v<=1e-14) continue;
        const key=region==='cap' && h===0 ? 'apex:0:0' :
          region==='cap' && h===90 ? 'side:'+az+':100' : region+':'+az+':'+h;
        stencil.set(key,(stencil.get(key)||0)+u*v);
      }
    }
    const b=Array(51).fill(0);
    for (const [key,value] of stencil) {
      const anchor=grid.get(key);
      if (!anchor) return null;
      for (let i=0; i<b.length; i++) b[i]+=value*anchor[i];
    }
    return [b,stencil.size===1?'direct':'engineering'];
  }

  function infer(region, phi, level) {
    if (!['side','cap'].includes(region) || !Number.isFinite(phi) || !Number.isFinite(level)) {
      throw new Error('Invalid surface coordinates');
    }
    const a=mod(phi,360)*Math.PI/180, beta=level*Math.PI/180;
    const radius=region==='side'?20:20*Math.sin(beta);
    const reference=[radius*Math.cos(a),radius*Math.sin(a),region==='side'?level:100+20*Math.cos(beta)];
    const value=fieldAt(region,mod(phi,360),level);
    if (!value) return {source:'unsupported',reference};
    const [field,source]=value;
    const paper=forward(field,weights.double), mlp=forward(field,weights.small), template=nearest(field);
    return {source,reference,paper,mlp,template,paper_mm:distance(reference,paper),
      mlp_mm:distance(reference,mlp),template_mm:distance(reference,template)};
  }

  global.MagTouchInfer=Object.freeze({infer});
}(globalThis));
