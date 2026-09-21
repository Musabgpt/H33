package com.musab.aragpt2;

/** Implement and register this interface to add any offline tool. */
public interface LocalTool {
    String name();
    String displayName();
    String description();
    String execute(String arguments) throws Exception;
}
