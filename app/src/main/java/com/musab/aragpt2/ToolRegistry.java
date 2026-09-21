package com.musab.aragpt2;

import android.content.Context;
import android.os.Build;
import java.time.Instant;
import java.util.*;

/** Lightweight tool extension point; no network permission is granted. */
public final class ToolRegistry {
    private final Map<String, LocalTool> tools = new LinkedHashMap<>();
    public static ToolRegistry withBuiltIns(Context ignored) {
        ToolRegistry r = new ToolRegistry();
        r.register(new LocalTool() {
            public String name(){return "device_info";} public String displayName(){return "معلومات الجهاز";}
            public String description(){return "Android version and device model; no arguments.";}
            public String execute(String args){return "Android "+Build.VERSION.RELEASE+", "+Build.MANUFACTURER+" "+Build.MODEL;}
        });
        r.register(new LocalTool() {
            public String name(){return "clock";} public String displayName(){return "الوقت";}
            public String description(){return "Current device time; no arguments.";}
            public String execute(String args){return Instant.now().toString();}
        });
        return r;
    }
    public synchronized void register(LocalTool tool) {
        if (tool == null || !tool.name().matches("[a-zA-Z0-9_.-]{1,48}")) throw new IllegalArgumentException("Invalid tool");
        tools.put(tool.name(), tool);
    }
    public synchronized LocalTool get(String name){return tools.get(name);}
    public synchronized int size(){return tools.size();}
    public synchronized List<LocalTool> snapshot(){return Collections.unmodifiableList(new ArrayList<>(tools.values()));}
    public synchronized String promptDescription(){
        if(tools.isEmpty()) return "";
        StringBuilder s=new StringBuilder("\nAvailable local tools:\n");
        for(LocalTool t:tools.values()) s.append("- ").append(t.name()).append(": ").append(t.description()).append('\n');
        s.append("Call one tool using exactly <tool_call name=\"TOOL\">ARGUMENTS</tool_call>.");
        return s.toString();
    }
}
