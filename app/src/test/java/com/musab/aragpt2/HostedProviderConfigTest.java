package com.musab.aragpt2;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class HostedProviderConfigTest {
    @Test
    public void blankKeyMeansUnconfigured() {
        HostedProviderConfig c =
                new HostedProviderConfig("gemini", "gemini-3.8-flash", "   ");
        assertFalse(c.isConfigured());
    }

    @Test
    public void geminiWithModelAndKeyIsConfigured() {
        HostedProviderConfig c =
                new HostedProviderConfig("gemini", "gemini-3.8-flash", "secret-key");
        assertTrue(c.isConfigured());
    }

    @Test
    public void wrongProviderDoesNotMasqueradeAsGemini() {
        HostedProviderConfig c =
                new HostedProviderConfig("other", "gemini-3.8-flash", "secret-key");
        assertFalse(c.isConfigured());
    }
}
