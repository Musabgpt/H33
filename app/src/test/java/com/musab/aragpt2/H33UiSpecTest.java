package com.musab.aragpt2;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public final class H33UiSpecTest {
    @Test
    public void classicFigmaPaletteAndGeometryAreStable() {
        assertEquals(0xFF0F0F0F, H33UiSpec.BACKGROUND);
        assertEquals(0xFF2B2B2B, H33UiSpec.PANEL);
        assertEquals(0xFFB170FF, H33UiSpec.ACCENT);
        assertEquals(76, H33UiSpec.COMPOSER_HEIGHT_DP);
        assertEquals(58, H33UiSpec.ACTION_SIZE_DP);
        assertTrue(H33UiSpec.USER_BUBBLE_WIDTH_FRACTION > H33UiSpec.ASSISTANT_BUBBLE_WIDTH_FRACTION - 0.13f);
    }
}
