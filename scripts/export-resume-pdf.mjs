import { chromium } from "playwright";

const baseUrl = process.env.BASE_URL || "http://127.0.0.1:4000";
const outputPath = process.env.OUTPUT_PATH || "resume.pdf";

const browser = await chromium.launch();
try {
  const page = await browser.newPage();

  await page.goto(`${baseUrl}/resume/`, {
    waitUntil: "networkidle",
  });

  await page.pdf({
    path: outputPath,
    format: "A4",
    printBackground: true,
    margin: {
      top: "12mm",
      right: "12mm",
      bottom: "12mm",
      left: "12mm",
    },
  });

  console.log(`${outputPath} exported successfully`);
} catch (error) {
  console.error("Failed to export resume PDF:", error);
  process.exitCode = 1;
} finally {
  await browser.close();
}
