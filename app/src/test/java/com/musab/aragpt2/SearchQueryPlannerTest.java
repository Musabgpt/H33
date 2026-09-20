package com.musab.aragpt2;

import org.junit.Test;

import java.lang.reflect.Method;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class SearchQueryPlannerTest {
    private static String focused(String query) {
        try {
            Class<?> type = Class.forName("com.musab.aragpt2.SearchQueryPlanner");
            Method method = type.getDeclaredMethod("focused", String.class);
            method.setAccessible(true);
            return (String) method.invoke(null, query);
        } catch (Exception ex) {
            throw new AssertionError("SearchQueryPlanner.focused is required", ex);
        }
    }

    @Test
    public void englishPlanDropsQuestionWordsAndKeepsTopic() {
        String planned = focused(
                "How many continents are there in the world and what are their names?"
        ).toLowerCase();

        assertTrue(planned.contains("continents"));
        assertTrue(planned.contains("world"));
        assertTrue(planned.contains("names"));
        assertFalse(planned.matches(".*\\bhow\\b.*"));
        assertFalse(planned.matches(".*\\bmany\\b.*"));
        assertFalse(planned.matches(".*\\bwhat\\b.*"));
        assertFalse(planned.matches(".*\\bare\\b.*"));
        assertFalse(planned.matches(".*\\bthere\\b.*"));
    }

    @Test
    public void arabicPlanDropsInterrogativesAndKeepsTopic() {
        String planned = focused("كم عدد قارات العالم وماهي أسماؤها؟");

        assertTrue(planned.contains("قارات"));
        assertTrue(planned.contains("العالم") || planned.contains("عالم"));
        assertFalse(planned.contains("كم"));
        assertFalse(planned.contains("ماهي"));
    }

    @Test
    public void planPreservesNumbersAndNamedEntities() {
        String planned = focused("من فاز بكأس العالم 2026 في نيويورك؟");

        assertTrue(planned.contains("2026"));
        assertTrue(planned.contains("نيويورك"));
    }
}
