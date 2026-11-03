import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import Icon from './Icon';
import { list, statusText } from './format';

const tones = { active:0x08bd8b, complete:0x279c79, waiting:0xb1c4c1, paused:0xd6a84b, failed:0xdc6470 };

export default function Workflow({ steps, selected, onSelect }) {
  const host=useRef(null), engine=useRef(null), latest=useRef({steps,selected,onSelect});
  latest.current={steps,selected,onSelect};
  const [paused,setPaused]=useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const pausedRef=useRef(paused); pausedRef.current=paused;
  const [unsupported,setUnsupported]=useState(false);
  const count=Math.max(list(steps).length,5);
  useEffect(() => {
    const media=window.matchMedia('(prefers-reduced-motion: reduce)');
    const change=event => { if (event.matches) setPaused(true); };
    media.addEventListener('change',change); return () => media.removeEventListener('change',change);
  },[]);
  useEffect(() => {
    const element=host.current;
    let renderer;
    try { renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'low-power'}); }
    catch { setUnsupported(true); return; }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,1.6));
    renderer.shadowMap.enabled=true; renderer.shadowMap.type=THREE.PCFShadowMap;
    renderer.outputColorSpace=THREE.SRGBColorSpace; renderer.toneMapping=THREE.ACESFilmicToneMapping; renderer.toneMappingExposure=1.35;
    renderer.domElement.setAttribute('aria-label','실제 단계 상태를 표시하는 대화형 3D 작업 흐름');
    renderer.domElement.setAttribute('role','img');
    element.appendChild(renderer.domElement);
    const scene=new THREE.Scene(), camera=new THREE.OrthographicCamera(-6,6,3,-3,.1,100);
    camera.position.set(3.2,5.5,12); camera.lookAt(0,.9,0);
    scene.add(new THREE.HemisphereLight(0xffffff,0xafd0bd,2.2));
    const light=new THREE.DirectionalLight(0xffffff,4.2); light.position.set(-3,8,5); light.castShadow=true;
    light.shadow.mapSize.set(1024,1024); Object.assign(light.shadow.camera,{left:-7,right:7,top:5,bottom:-5}); light.shadow.bias=-.001; light.shadow.normalBias=.025; scene.add(light);
    const fill=new THREE.DirectionalLight(0xdcfff1,2); fill.position.set(4,2,-4); scene.add(fill);
    const material=color => new THREE.MeshStandardMaterial({color,roughness:.3,metalness:.16});
    const porcelain=material(0xf8fffc), white=material(0xffffff), trim=material(0xc7dcd4), dark=material(0x273f48), face=material(0x9bdcc0);
    const group=new THREE.Group(); scene.add(group);
    const box=(parent,x,y,z,w,h,d,mat,r=.08) => { const mesh=new THREE.Mesh(new RoundedBoxGeometry(w,h,d,3,r),mat); mesh.position.set(x,y,z); mesh.castShadow=true; mesh.receiveShadow=true; parent.add(mesh); return mesh; };
    const cylinder=(parent,x,y,z,r,h,mat) => { const m=new THREE.Mesh(new THREE.CylinderGeometry(r,r,h,32),mat); m.position.set(x,y,z); m.castShadow=true; parent.add(m); return m; };
    const sphere=(parent,x,y,z,r,mat) => {const m=new THREE.Mesh(new THREE.SphereGeometry(r,28,18),mat);m.position.set(x,y,z);m.castShadow=true;parent.add(m);return m;};
    const width=count*1.7+.4;
    box(group,0,.13,0,width,.3,2.05,porcelain,.14);
    box(group,0,-.08,0,width-.3,.16,1.8,trim,.06);
    // 원통 연결관은 모형의 구조이며 실제 진행률을 표현하지 않습니다.
    const machines=[];
    for(let i=0;i<count;i++) {
      const node=new THREE.Group();node.position.x=(i-(count-1)/2)*1.7;group.add(node);node.userData.stageIndex=i;
      const glow=material(tones.waiting);glow.emissive.setHex(tones.waiting);glow.emissiveIntensity=.08;
      box(node,0,.4,0,1.44,.26,1.55,white,.14);
      const rim=box(node,0,.51,0,1.39,.05,1.49,glow,.12);
      box(node,0,.94,0,1.18,.88,1.16,porcelain,.14);
      box(node,0,.76,.604,.71,.12,.022,glow,.035);
      box(node,0,.56,.7,.6,.12,.42,white,.04);
      if(i<count-1){const pipe=cylinder(node,.85,.94,0,.105,.58,trim);pipe.rotation.z=Math.PI/2;const inside=cylinder(node,.85,.94,0,.065,.6,glow);inside.rotation.z=Math.PI/2;}
      const ornament=new THREE.Group();node.add(ornament);
      const fans=[];
      if(i===0){
        sphere(ornament,-.28,1.75,0,.34,white);sphere(ornament,.03,1.94,0,.43,white);sphere(ornament,.36,1.75,0,.31,white);
        box(ornament,0,1.66,0,.9,.3,.5,white,.14);
        const shaft=cylinder(ornament,0,1.55,.48,.07,.47,glow);
        const arrow=new THREE.Mesh(new THREE.ConeGeometry(.19,.23,4),glow);arrow.position.set(0,1.28,.48);arrow.rotation.z=Math.PI;ornament.add(arrow);
        for(let k=0;k<4;k++)box(ornament,(k%2-.5)*.16,1.05+Math.floor(k/2)*.13,.55,.08,.08,.08,glow,.01);
        shaft.userData.decorative=true;
      } else if(i===1){
        for(let k=0;k<3;k++){
          const photo=new THREE.Group();photo.position.set((k-1)*.14,1.65,-.12+k*.13);photo.rotation.z=-.12+k*.06;
          box(photo,0,0,0,.82,.95,.07,white,.05);box(photo,0,0,.05,.67,.77,.025,face,.02);
          sphere(photo,.18,.21,.08,.075,white);
          const mountain=new THREE.Mesh(new THREE.ConeGeometry(.3,.35,3),white);mountain.scale.z=.08;mountain.position.set(-.05,-.16,.09);photo.add(mountain);ornament.add(photo);
        }
      } else if(i===2){
        box(ornament,0,1.75,0,1.02,.75,.22,dark,.06);box(ornament,0,1.77,-.06,1.1,.86,.05,trim,.025);
        for(let k=0;k<7;k++)box(ornament,-.43+k*.14,2.22,-.04,.07,.17,.17,trim,.01);
        const fan=new THREE.Group();fan.position.set(.16,1.75,.15);ornament.add(fan);
        const ring=new THREE.Mesh(new THREE.TorusGeometry(.25,.033,12,48),trim);fan.add(ring);
        sphere(fan,0,0,.015,.075,trim);
        for(let j=0;j<7;j++){const blade=box(fan,Math.cos(j*Math.PI*2/7)*.14,Math.sin(j*Math.PI*2/7)*.14,0,.15,.07,.035,trim,.025);blade.rotation.z=j*Math.PI*2/7+.5;}
        fans.push(fan);
        box(ornament,-.37,1.75,.14,.08,.35,.025,face,.015);
      } else if(i===3){
        const lens=new THREE.Mesh(new THREE.TorusGeometry(.42,.075,20,64),trim);lens.position.set(-.05,1.94,0);ornament.add(lens);
        const inner=new THREE.Mesh(new THREE.CircleGeometry(.355,48),new THREE.MeshPhysicalMaterial({color:0xc3f5df,transparent:true,opacity:.6,roughness:.1,metalness:.1,side:THREE.DoubleSide}));inner.position.set(-.05,1.94,-.01);ornament.add(inner);
        const handle=box(ornament,.25,1.48,0,.16,.52,.16,dark,.05);handle.rotation.z=.55;
        for(let k=0;k<4;k++)sphere(node,-.3+k*.2,1.1,.6,.033,glow);
      } else {
        const paper=box(ornament,0,1.83,0,.92,1.3,.1,white,.055);paper.rotation.x=-.12;
        for(let k=0;k<4;k++)box(ornament,-.08,2.24-k*.15,.075,.55-k*.075,.04,.025,trim,.012);
        for(let k=0;k<3;k++)box(ornament,-.2+k*.2,1.49+k*.055,.1,.09,.15+k*.11,.03,face,.012);
      }
      const beacon=cylinder(node,0,1.47,-.35,.075,.13,glow);
      machines.push({node,ornament,glow,rim,beacon,fans});
    }
    const plane=new THREE.Mesh(new THREE.PlaneGeometry(50,50),new THREE.ShadowMaterial({opacity:.13}));plane.rotation.x=-Math.PI/2;plane.position.y=-.18;plane.receiveShadow=true;scene.add(plane);
    const resize=()=>{const w=element.clientWidth,h=element.clientHeight;if(!w||!h)return;renderer.setSize(w,h);const aspect=w/h;const halfW=width*.62;camera.left=-halfW;camera.right=halfW;camera.top=halfW/aspect;camera.bottom=-halfW/aspect;camera.updateProjectionMatrix();};
    const observer=new ResizeObserver(resize);observer.observe(element);resize();
    const raycaster=new THREE.Raycaster();
    const click=event=>{const rect=renderer.domElement.getBoundingClientRect();raycaster.setFromCamera(new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1),camera);const hits=raycaster.intersectObjects(group.children,true);if(!hits.length)return;let node=hits[0].object;while(node&&!Number.isInteger(node.userData.stageIndex))node=node.parent;if(node&&latest.current.steps[node.userData.stageIndex])latest.current.onSelect(node.userData.stageIndex);};
    renderer.domElement.addEventListener('click',click);
    let animationTime=0,last=0,frame=0;
    renderer.setAnimationLoop(time=>{
      const delta=last?Math.min((time-last)/1000,.05):0;last=time;
      if(!pausedRef.current)animationTime+=delta;
      for(let i=0;i<machines.length;i++){
        const m=machines[i],status=latest.current.steps[i]?.status||'waiting',color=tones[status]||tones.waiting;
        m.glow.color.setHex(color);m.glow.emissive.setHex(color);
        m.glow.emissiveIntensity=status==='active' ? .3+Math.sin(animationTime*2.2)*.18 : .05;
        m.node.position.y=latest.current.selected===i ? .035:0;
        m.ornament.position.y=pausedRef.current?0:Math.sin(animationTime*1.2+i*.7)*.025;
        for(const fan of m.fans)fan.rotation.z=-animationTime*(status==='active'?2.3:.3);
      }
      renderer.render(scene,camera);
      renderer.domElement.dataset.motionFrame=String(pausedRef.current?frame:++frame);
      renderer.domElement.dataset.motion=pausedRef.current?'paused':'playing';
    });
    engine.current=renderer;
    return()=>{observer.disconnect();renderer.domElement.removeEventListener('click',click);renderer.setAnimationLoop(null);const geometries=new Set(),materials=new Set();scene.traverse(object=>{if(object.geometry)geometries.add(object.geometry);if(object.material)materials.add(object.material);});geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());renderer.dispose();renderer.domElement.remove();engine.current=null;};
  },[count]);
  return <div className="workflow">
    <div className="motion-controls"><button onClick={()=>setPaused(!paused)} aria-pressed={paused}><Icon name={paused?'play':'pause'} size={17}/>{paused?'3D 재생':'3D 일시정지'}</button></div>
    <div ref={host} className="scene-host" data-testid="workflow-scene" />
    {unsupported&&<p className="webgl-error" role="status">이 브라우저에서 WebGL을 사용할 수 없습니다. 아래 단계에서 실제 상태를 확인하세요.</p>}
    <div className="stage-buttons">{list(steps).length?steps.map((step,index)=><button key={step.id} className={`stage-button ${selected===index?'selected':''}`} onClick={()=>onSelect(index)} aria-pressed={selected===index}><span className={`status-dot ${step.status}`}/><span>{step.title}<small>{statusText(step.status)}</small></span></button>):<p className="muted">실제 단계 정보를 기다리고 있습니다.</p>}</div>
    <p className="workflow-caption">입체 모형 또는 단계를 선택해 설명을 확인하세요. 선택은 작업을 실행하지 않습니다.</p>
  </div>;
}
