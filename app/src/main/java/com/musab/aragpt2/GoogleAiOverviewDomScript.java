package com.musab.aragpt2;

public final class GoogleAiOverviewDomScript {
    private GoogleAiOverviewDomScript() {}

    public static String buildExtractionScript() {
        return "(function(){"
                + "const markers=['نبذة باستخدام الذكاء الاصطناعي','AI Overview'];"
                + "const uiOnly=new Set(['عرض المزيد','Show more','المزيد','More']);"
                + "const norm=s=>(s||'').replace(/\\u00a0/g,' ').replace(/[ \\t]+/g,' ').trim();"
                + "const visible=e=>{if(!e)return false;const s=getComputedStyle(e);"
                + "return s.display!=='none'&&s.visibility!=='hidden'&&e.getClientRects().length>0;};"
                + "const nodes=Array.from(document.querySelectorAll('body *'));"
                + "const markerNodes=[];"
                + "for(const e of nodes){if(!visible(e))continue;"
                + "const t=norm(e.innerText||e.textContent);"
                + "if(markers.some(m=>t===m||t.startsWith(m)))markerNodes.push(e);}"
                + "if(!markerNodes.length)return JSON.stringify({found:false,text:'',sources:[]});"
                + "markerNodes.sort((a,b)=>norm(a.innerText||a.textContent).length-norm(b.innerText||b.textContent).length);"
                + "let marker=markerNodes[0];"
                + "let candidate=null,cur=marker;"
                + "while(cur&&cur!==document.body){"
                + "const t=norm(cur.innerText||cur.textContent);"
                + "if(t.length>=60&&t.length<=8000&&markers.some(m=>t.includes(m))){"
                + "candidate=cur;break;}cur=cur.parentElement;}"
                + "if(!candidate)return JSON.stringify({found:false,text:'',sources:[]});"
                + "const raw=(candidate.innerText||candidate.textContent||'');"
                + "const lines=raw.split(/\\n+/).map(norm).filter(Boolean).filter(line=>"
                + "!markers.includes(line)&&!uiOnly.has(line));"
                + "const text=lines.join('\\n').trim();"
                + "if(text.length<20)return JSON.stringify({found:false,text:'',sources:[]});"
                + "const sources=[];const seen=new Set();"
                + "for(const a of candidate.querySelectorAll('a[href]')){"
                + "let href='';try{href=new URL(a.href,location.href).href;}catch(e){}"
                + "if(!/^https?:\\/\\//i.test(href)||seen.has(href))continue;"
                + "seen.add(href);"
                + "const title=norm(a.innerText||a.getAttribute('aria-label')||a.title||'');"
                + "sources.push({title:title,url:href});}"
                + "return JSON.stringify({found:true,text:text,sources:sources});"
                + "})()";
    }
}
