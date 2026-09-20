package com.musab.aragpt2;

import android.app.Activity;
import android.graphics.Bitmap;
import android.net.Uri;
import android.net.http.SslError;
import android.os.Handler;
import android.os.Looper;
import android.view.ViewGroup;
import android.webkit.CookieManager;
import android.webkit.SslErrorHandler;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

public final class GoogleAiOverviewProvider implements HostedAnswerProvider {
    private static final long OVERALL_TIMEOUT_MS = 10_000L;
    private static final long POLL_INTERVAL_MS = 650L;

    private final Activity activity;
    private final Handler mainHandler;

    public GoogleAiOverviewProvider(Activity activity) {
        if (activity == null) throw new IllegalArgumentException("activity required");
        this.activity = activity;
        this.mainHandler = new Handler(Looper.getMainLooper());
    }

    @Override
    public AnswerCandidate answer(String question) throws Exception {
        String q = clean(question);
        if (q.isEmpty()) {
            return unavailable("Google AI Overview غير متاح لسؤال فارغ");
        }
        if (Looper.myLooper() == Looper.getMainLooper()) {
            throw new IllegalStateException(
                    "Google AI Overview يجب أن يعمل خارج واجهة المستخدم");
        }
        if (activity.isFinishing() || activity.isDestroyed()) {
            return unavailable("Google AI Overview غير متاح الآن");
        }

        String url = buildSearchUrl(q, "ar", "GB");
        CountDownLatch latch = new CountDownLatch(1);
        AtomicReference<AnswerCandidate> result =
                new AtomicReference<>(unavailable(
                        "Google AI Overview غير متاح لهذا البحث"));
        AtomicBoolean completed = new AtomicBoolean(false);
        AtomicReference<WebView> webViewRef = new AtomicReference<>();

        mainHandler.post(() -> {
            if (activity.isFinishing() || activity.isDestroyed()) {
                complete(completed, latch, result, webViewRef,
                        unavailable("Google AI Overview غير متاح الآن"));
                return;
            }

            try {
                WebView webView = new WebView(activity);
                webViewRef.set(webView);
                configure(webView);

                ViewGroup root =
                        (ViewGroup) activity.findViewById(android.R.id.content);
                ViewGroup.LayoutParams params =
                        new ViewGroup.LayoutParams(
                                ViewGroup.LayoutParams.MATCH_PARENT,
                                ViewGroup.LayoutParams.MATCH_PARENT);
                root.addView(webView, params);
                webView.setVisibility(android.view.View.INVISIBLE);

                long deadline = android.os.SystemClock.uptimeMillis()
                        + OVERALL_TIMEOUT_MS;

                Runnable timeout = () -> complete(
                        completed, latch, result, webViewRef,
                        unavailable("Google AI Overview غير متاح لهذا البحث"));
                mainHandler.postDelayed(timeout, OVERALL_TIMEOUT_MS);

                webView.setWebViewClient(new WebViewClient() {
                    @Override
                    public boolean shouldOverrideUrlLoading(
                            WebView view, WebResourceRequest request) {
                        Uri target = request.getUrl();
                        String host = target == null ? "" : clean(target.getHost());
                        return !isAllowedGoogleHost(host);
                    }

                    @Override
                    public void onPageStarted(
                            WebView view, String pageUrl, Bitmap favicon) {
                        if (isBlockedGooglePage(pageUrl)) {
                            mainHandler.removeCallbacks(timeout);
                            complete(completed, latch, result, webViewRef,
                                    unavailable(
                                            "Google يطلب موافقة أو تحقق لهذا البحث"));
                        }
                    }

                    @Override
                    public void onPageFinished(WebView view, String pageUrl) {
                        if (completed.get()) return;
                        if (isBlockedGooglePage(pageUrl)) {
                            mainHandler.removeCallbacks(timeout);
                            complete(completed, latch, result, webViewRef,
                                    unavailable(
                                            "Google يطلب موافقة أو تحقق لهذا البحث"));
                            return;
                        }
                        pollOverview(
                                view, deadline, timeout,
                                completed, latch, result, webViewRef);
                    }

                    @Override
                    public void onReceivedError(
                            WebView view,
                            WebResourceRequest request,
                            WebResourceError error) {
                        if (request != null && request.isForMainFrame()) {
                            mainHandler.removeCallbacks(timeout);
                            complete(completed, latch, result, webViewRef,
                                    unavailable(
                                            "تعذر تحميل صفحة Google Search"));
                        }
                    }

                    @Override
                    public void onReceivedSslError(
                            WebView view,
                            SslErrorHandler handler,
                            SslError error) {
                        if (handler != null) handler.cancel();
                        mainHandler.removeCallbacks(timeout);
                        complete(completed, latch, result, webViewRef,
                                unavailable("فشل اتصال Google الآمن"));
                    }
                });

                webView.loadUrl(url);
            } catch (Exception ex) {
                complete(completed, latch, result, webViewRef,
                        unavailable("تعذر تشغيل Google AI Overview"));
            }
        });

        boolean finished = latch.await(
                OVERALL_TIMEOUT_MS + 2_000L, TimeUnit.MILLISECONDS);
        if (!finished) {
            mainHandler.post(() -> complete(
                    completed, latch, result, webViewRef,
                    unavailable("انتهت مهلة Google AI Overview")));
        }
        return result.get();
    }

    static String buildSearchUrl(String question, String language, String country) {
        String q = clean(question);
        String hl = clean(language);
        String gl = clean(country);
        if (hl.isEmpty()) hl = "ar";
        if (gl.isEmpty()) gl = "GB";

        String encoded = encode(q);
        return "https://www.google.com/search?q=" + encoded
                + "&hl=" + encode(hl)
                + "&gl=" + encode(gl);
    }

    private void configure(WebView webView) {
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setAllowFileAccess(false);
        settings.setAllowContentAccess(false);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        settings.setBlockNetworkImage(true);
        settings.setJavaScriptCanOpenWindowsAutomatically(false);
        settings.setSupportMultipleWindows(false);
        settings.setUserAgentString(
                chromeLikeUserAgent(WebSettings.getDefaultUserAgent(activity)));

        CookieManager cookieManager = CookieManager.getInstance();
        cookieManager.setAcceptCookie(true);
        cookieManager.setAcceptThirdPartyCookies(webView, false);
    }

    private void pollOverview(
            WebView webView,
            long deadline,
            Runnable timeout,
            AtomicBoolean completed,
            CountDownLatch latch,
            AtomicReference<AnswerCandidate> result,
            AtomicReference<WebView> webViewRef) {

        if (completed.get()) return;
        if (android.os.SystemClock.uptimeMillis() >= deadline) return;

        webView.evaluateJavascript(
                GoogleAiOverviewDomScript.buildExtractionScript(),
                raw -> {
                    if (completed.get()) return;

                    try {
                        GoogleAiOverviewParser.Result parsed =
                                GoogleAiOverviewParser
                                        .parseEvaluateJavascriptResult(raw);
                        if (parsed.found) {
                            mainHandler.removeCallbacks(timeout);
                            complete(
                                    completed, latch, result, webViewRef,
                                    AnswerCandidate.available(
                                            "hosted",
                                            AnswerCandidate.Kind.HOSTED,
                                            "google-ai-overview",
                                            parsed.text,
                                            parsed.sources,
                                            "Google AI Overview"
                                    )
                            );
                            return;
                        }
                    } catch (Exception ignored) {
                        // Dynamic Google DOM can be incomplete while loading;
                        // keep polling until the bounded deadline.
                    }

                    long remaining = deadline
                            - android.os.SystemClock.uptimeMillis();
                    if (remaining > 0 && !completed.get()) {
                        mainHandler.postDelayed(
                                () -> pollOverview(
                                        webView, deadline, timeout,
                                        completed, latch, result, webViewRef),
                                Math.min(POLL_INTERVAL_MS, remaining)
                        );
                    }
                });
    }

    private void complete(
            AtomicBoolean completed,
            CountDownLatch latch,
            AtomicReference<AnswerCandidate> result,
            AtomicReference<WebView> webViewRef,
            AnswerCandidate answer) {

        if (!completed.compareAndSet(false, true)) return;
        result.set(answer == null
                ? unavailable("Google AI Overview غير متاح لهذا البحث")
                : answer);

        WebView webView = webViewRef.getAndSet(null);
        if (webView != null) {
            try {
                webView.stopLoading();
                webView.setWebViewClient(null);
                ViewGroup parent = (ViewGroup) webView.getParent();
                if (parent != null) parent.removeView(webView);
                webView.loadUrl("about:blank");
                webView.clearHistory();
                webView.removeAllViews();
                webView.destroy();
            } catch (Exception ignored) {
                // Result is already fixed; cleanup remains best-effort.
            }
        }
        latch.countDown();
    }

    static String chromeLikeUserAgent(String value) {
        String ua = clean(value);
        if (ua.isEmpty()) return ua;

        ua = ua.replace("; wv)", ")");
        ua = ua.replace("; wv;", ";");
        ua = ua.replace(" wv)", ")");
        ua = ua.replace("Version/4.0 ", "");
        ua = ua.replaceAll("\\s{2,}", " ").trim();
        return ua;
    }

    private static String encode(String value) {
        try {
            return URLEncoder.encode(
                    clean(value), StandardCharsets.UTF_8.name())
                    .replace("+", "%20");
        } catch (Exception ex) {
            throw new IllegalStateException("تعذر ترميز رابط Google", ex);
        }
    }

    static boolean isAllowedGoogleHost(String value) {
        String host = clean(value).toLowerCase(java.util.Locale.ROOT);
        if (host.startsWith("www.")) host = host.substring(4);
        return host.equals("google.com")
                || host.endsWith(".google.com")
                || host.equals("google.co.uk")
                || host.endsWith(".google.co.uk");
    }

    private static boolean isBlockedGooglePage(String value) {
        String url = clean(value).toLowerCase();
        return url.contains("consent.google.")
                || url.contains("/sorry/")
                || url.contains("recaptcha")
                || url.contains("captcha");
    }

    private static AnswerCandidate unavailable(String status) {
        return AnswerCandidate.unavailable(
                "hosted",
                AnswerCandidate.Kind.HOSTED,
                "google-ai-overview",
                status
        );
    }

    private static String clean(String value) {
        return value == null ? "" : value.trim();
    }
}
