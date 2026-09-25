import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

public class Main {

    private static final String API_URL = "https://api.anthropic.com/v1/messages";
    private static final String MODEL = "claude-sonnet-4-6";
    private static final String DATA_FOLDER = "data";
    private static final int CHUNK_SIZE = 1500;
    private static final int TOP_CHUNKS = 6;
    private static final int MAX_TOKENS = 2048;

    record Chunk(String source, String text) {}

    public static void main(String[] args) throws Exception {
        String apiKey = System.getenv("ANTHROPIC_API_KEY");
        if (apiKey == null || apiKey.isEmpty()) {
            System.out.println("No API key found.");
            System.out.println("Set the ANTHROPIC_API_KEY environment variable, then run again.");
            return;
        }

        List<Chunk> chunks = loadAndChunkData();
        if (chunks.isEmpty()) {
            System.out.println("No files found in the '" + DATA_FOLDER + "' folder. Continuing without data.\n");
        } else {
            System.out.println("Loaded " + chunks.size() + " chunk(s) of data from '" + DATA_FOLDER + "'.\n");
        }

        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(20))
                .build();

        System.out.println("Claude Agent ready. Type a question (or 'exit' to quit).");

        while (true) {
            System.out.print("\nYou: ");
            String prompt = reader.readLine();
            if (prompt == null || prompt.trim().equalsIgnoreCase("exit")) {
                System.out.println("Bye.");
                break;
            }
            if (prompt.trim().isEmpty()) {
                continue;
            }

            try {
                List<Chunk> relevant = findRelevantChunks(chunks, prompt);
                String reply = callClaude(client, prompt, relevant, apiKey);
                System.out.println("Claude: " + reply);
            } catch (Exception ex) {
                System.out.println("Error: " + ex.getMessage());
            }
        }
    }

    private static List<Chunk> loadAndChunkData() {
        List<Chunk> chunks = new ArrayList<>();
        Path dataDir = Path.of(DATA_FOLDER);
        if (!Files.isDirectory(dataDir)) {
            return chunks;
        }

        try (Stream<Path> files = Files.walk(dataDir)) {
            List<Path> txtFiles = files
                    .filter(Files::isRegularFile)
                    .filter(p -> p.toString().toLowerCase().endsWith(".txt"))
                    .sorted()
                    .toList();

            for (Path file : txtFiles) {
                String content = Files.readString(file, StandardCharsets.UTF_8);
                String fileName = file.getFileName().toString();
                for (int i = 0; i < content.length(); i += CHUNK_SIZE) {
                    int end = Math.min(i + CHUNK_SIZE, content.length());
                    chunks.add(new Chunk(fileName, content.substring(i, end)));
                }
            }
        } catch (IOException e) {
            System.out.println("Warning: couldn't read data folder: " + e.getMessage());
        }
        return chunks;
    }

    private static List<Chunk> findRelevantChunks(List<Chunk> chunks, String query) {
        if (chunks.isEmpty()) {
            return chunks;
        }

        List<String> queryWords = new ArrayList<>();
        for (String w : query.toLowerCase().split("\\W+")) {
            if (w.length() > 2) {
                queryWords.add(w);
            }
        }

        List<Chunk> scored = new ArrayList<>(chunks);
        scored.sort(Comparator.comparingInt((Chunk c) -> score(c, queryWords)).reversed());

        int limit = Math.min(TOP_CHUNKS, scored.size());
        return scored.subList(0, limit);
    }

    private static int score(Chunk chunk, List<String> queryWords) {
        String lowerText = chunk.text().toLowerCase();
        int total = 0;
        for (String word : queryWords) {
            int idx = 0;
            while ((idx = lowerText.indexOf(word, idx)) != -1) {
                total++;
                idx += word.length();
            }
        }
        return total;
    }

    private static String callClaude(HttpClient client, String prompt, List<Chunk> relevantChunks, String apiKey) throws Exception {
        StringBuilder context = new StringBuilder();
        for (Chunk c : relevantChunks) {
            context.append("=== From: ").append(c.source()).append(" ===\n");
            context.append(c.text()).append("\n\n");
        }

        String fullPrompt = context.isEmpty()
                ? prompt
                : "Here are the most relevant excerpts from the reference data for this question:\n\n"
                  + context + "\nUsing the excerpts above where relevant, answer this question:\n" + prompt;

        String escapedPrompt = fullPrompt
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n");

        String body = "{"
                + "\"model\":\"" + MODEL + "\","
                + "\"max_tokens\":" + MAX_TOKENS + ","
                + "\"messages\":[{\"role\":\"user\",\"content\":\"" + escapedPrompt + "\"}]"
                + "}";

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(API_URL))
                .timeout(Duration.ofSeconds(60))
                .header("Content-Type", "application/json")
                .header("x-api-key", apiKey)
                .header("anthropic-version", "2023-06-01")
                .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
                .build();

        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            return "API error (" + response.statusCode() + "): " + response.body();
        }
        return extractText(response.body());
    }

    private static String extractText(String json) {
        StringBuilder result = new StringBuilder();
        Pattern pattern = Pattern.compile("\"text\"\\s*:\\s*\"((?:\\\\.|[^\"\\\\])*)\"");
        Matcher matcher = pattern.matcher(json);
        while (matcher.find()) {
            result.append(matcher.group(1)
                    .replace("\\n", "\n")
                    .replace("\\\"", "\"")
                    .replace("\\\\", "\\"));
        }
        return result.length() > 0 ? result.toString() : "No text found in response:\n" + json;
    }
}
