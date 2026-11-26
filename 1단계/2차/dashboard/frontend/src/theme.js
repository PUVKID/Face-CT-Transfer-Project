// 테마 선택만 저장하며 실험 상태·설정에는 쓰지 않습니다.
const KEY='face-benchmark-theme';
export function getTheme(){return document.documentElement.dataset.theme==='dark'?'dark':'light';}
export function setTheme(value){const theme=value==='dark'?'dark':'light';document.documentElement.dataset.theme=theme;try{localStorage.setItem(KEY,theme);return true;}catch{return false;}}
let initial='light';try{initial=localStorage.getItem(KEY)==='dark'?'dark':'light';}catch{}
document.documentElement.dataset.theme=initial;
