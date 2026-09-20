package com.musab.aragpt2;

import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class ProviderSecretStoreTest {
    @Test
    public void savesEncryptedKeyAndRestoresConfig() throws Exception {
        FakeBackend backend = new FakeBackend();
        FakeCipher cipher = new FakeCipher();
        ProviderSecretStore store = new ProviderSecretStore(backend, cipher);

        store.saveGemini("my-secret-key", "gemini-3.8-flash");

        assertFalse(backend.values.containsValue("my-secret-key"));
        assertEquals("ciphertext", backend.values.get("gemini_key_ciphertext"));
        assertEquals("iv", backend.values.get("gemini_key_iv"));
        assertEquals("gemini-3.8-flash", backend.values.get("gemini_model"));

        HostedProviderConfig config = store.loadGeminiConfig();
        assertTrue(config.isConfigured());
        assertEquals("my-secret-key", config.apiKey);
        assertEquals("gemini-3.8-flash", config.model);
    }

    @Test
    public void clearRemovesCredentialsButDefaultModelRemainsAvailable() throws Exception {
        FakeBackend backend = new FakeBackend();
        FakeCipher cipher = new FakeCipher();
        ProviderSecretStore store = new ProviderSecretStore(backend, cipher);

        store.saveGemini("my-secret-key", "gemini-3.8-flash");
        store.clearGemini();

        assertFalse(store.hasGeminiKey());
        HostedProviderConfig config = store.loadGeminiConfig();
        assertFalse(config.isConfigured());
        assertEquals(HostedProviderConfig.DEFAULT_GEMINI_MODEL, config.model);
    }

    @Test(expected = IllegalArgumentException.class)
    public void refusesBlankApiKey() throws Exception {
        ProviderSecretStore store =
                new ProviderSecretStore(new FakeBackend(), new FakeCipher());
        store.saveGemini("   ", "gemini-3.8-flash");
    }

    private static final class FakeBackend implements ProviderSecretStore.Backend {
        final Map<String, String> values = new HashMap<>();

        @Override
        public String get(String key) {
            String value = values.get(key);
            return value == null ? "" : value;
        }

        @Override
        public void put(String key, String value) {
            values.put(key, value);
        }

        @Override
        public void remove(String key) {
            values.remove(key);
        }
    }

    private static final class FakeCipher implements ProviderSecretStore.CipherBox {
        private String lastPlaintext = "";

        @Override
        public ProviderSecretStore.EncryptedSecret encrypt(String plaintext) {
            lastPlaintext = plaintext;
            return new ProviderSecretStore.EncryptedSecret("ciphertext", "iv");
        }

        @Override
        public String decrypt(String ciphertext, String iv) {
            return lastPlaintext;
        }
    }
}
