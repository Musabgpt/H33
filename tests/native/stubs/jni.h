#pragma once
#include <memory>
#include <string>
#include <vector>
#include <cstdarg>
#define JNIEXPORT
#define JNICALL
using jint = int;
using jsize = int;
using jbyte = signed char;
struct JavaValue { std::string text; };
using jobject = JavaValue *;
using jstring = jobject;
using jclass = jobject;
using jbyteArray = jobject;
using jmethodID = const char *;
// Only the JNI string/callback boundary is replaced; tests include the actual
// production translation unit and run its open/generate/cancel entry points.
struct JNIEnv {
    std::vector<std::unique_ptr<JavaValue>> refs;
    std::string error;
    jobject make(std::string value = {}) {
        refs.emplace_back(new JavaValue{value}); return refs.back().get();
    }
    const char * GetStringUTFChars(jstring s, void *) { return s->text.c_str(); }
    void ReleaseStringUTFChars(jstring, const char *) {}
    jbyteArray NewByteArray(jsize n) { return make(std::string(n, '\0')); }
    void SetByteArrayRegion(jbyteArray b, jsize start, jsize n, const jbyte * data) {
        b->text.replace(start, n, reinterpret_cast<const char *>(data), n);
    }
    jclass FindClass(const char * name) { return make(name); }
    jmethodID GetMethodID(jclass, const char * name, const char *) { return name; }
    jstring NewStringUTF(const char * text) { return make(text); }
    jobject NewObject(jclass, jmethodID, jbyteArray bytes, jstring) { return make(bytes->text); }
    void DeleteLocalRef(jobject) {}
    jclass GetObjectClass(jobject) { return make(); }
    void CallVoidMethod(jobject, jmethodID, jstring) {}
    bool ExceptionCheck() { return !error.empty(); }
    void ExceptionClear() { error.clear(); }
    int ThrowNew(jclass, const char * message) { error = message; return 0; }
};
