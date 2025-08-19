import pg from 'pg';
const { Pool } = pg;
import express from 'express';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';
import { WebSocketServer } from 'ws';
import fs from 'fs';
import https from 'https';
import puppeteer from 'puppeteer-extra';
import stealthPlugin from 'puppeteer-extra-plugin-stealth';
puppeteer.use(stealthPlugin());
import multer from 'multer';
import OpenAI from "openai";
import { Server } from 'socket.io';
import dotenv from 'dotenv';
dotenv.config();
import { spawn } from 'child_process';
import bodyParser from 'body-parser';
import pLimit from 'p-limit';
process.env['NODE_TLS_REJECT_UNAUTHORIZED'] = 0;
const pool = new Pool({
  user: 'postgres',
  host: '127.0.0.1',
  database: 'postgres',
  password: 'postgres',
  port: 5432,
});
async function createTablesForProject(project) {
  const client = await pool.connect();
  const query = `
    -- Node table (adapted from Python)
    CREATE TABLE IF NOT EXISTS "${project}_nodes" (
      id SERIAL PRIMARY KEY,
      s_id TEXT UNIQUE, -- URL or key
      type TEXT, -- e.g., 'person', 'image', 'video', 'domain', 'profile'
      content JSONB -- Flexible data (name, details, etc.)
    );
    -- Edge table (adapted from Python, replaces connections)
    CREATE TABLE IF NOT EXISTS "${project}_edges" (
      id SERIAL PRIMARY KEY,
      from_id INTEGER REFERENCES "${project}_nodes"(id),
      to_id INTEGER REFERENCES "${project}_nodes"(id),
      type TEXT DEFAULT 'connection',
      UNIQUE (from_id, to_id, type)
    );
    -- Optional: Legacy tables if needed, but we'll migrate to nodes/edges
    -- DROP TABLE IF EXISTS "${project}_domainConfigs";
    -- DROP TABLE IF EXISTS "${project}_persons";
    -- DROP TABLE IF EXISTS "${project}_connections";
  `;
  const alterQuery = `
    ALTER TABLE "${project}_nodes" ADD COLUMN IF NOT EXISTS page_text TEXT;
  `;
  try {
    await client.query(query);
    await client.query(alterQuery);
  } catch (error) {
    console.error('An error occurred:', error);
  } finally {
    client.release();
  }
}
function deepMerge(target, source) {
  if (typeof target !== 'object' || typeof source !== 'object' || target === null || source === null) {
    return source;
  }
  if (Array.isArray(target) && Array.isArray(source)) {
    return [...target, ...source];
  }
  const merged = { ...target };
  for (const key in source) {
    if (source.hasOwnProperty(key)) {
      if (typeof source[key] === 'object' && source[key] !== null) {
        merged[key] = deepMerge(merged[key] || (Array.isArray(source[key]) ? [] : {}), source[key]);
      } else {
        merged[key] = source[key];
      }
    }
  }
  return merged;
}
async function insertNode(s_id, type, content, project, page_text = null) {
  const client = await pool.connect();
  try {
    // Check if exists
    const selectQuery = `SELECT id, content, type FROM "${project}_nodes" WHERE s_id = $1`;
    const selectResult = await client.query(selectQuery, [s_id]);
    if (selectResult.rows.length > 0) {
      // Exists, deep merge content and update
      const existing = selectResult.rows[0];
      const newContent = deepMerge(existing.content, content);
      const newType = type; // Update to new type
      const updateQuery = `
        UPDATE "${project}_nodes"
        SET content = $1, type = $2, page_text = $4
        WHERE id = $3
        RETURNING id;
      `;
      const updateValues = [newContent, newType, existing.id, page_text];
      const updateResult = await client.query(updateQuery, updateValues);
      return updateResult.rows[0].id;
    } else {
      // Insert new
      const insertQuery = `
        INSERT INTO "${project}_nodes" (s_id, type, content, page_text)
        VALUES ($1, $2, $3, $4)
        RETURNING id;
      `;
      const insertValues = [s_id, type, content, page_text];
      const insertResult = await client.query(insertQuery, insertValues);
      return insertResult.rows[0].id;
    }
  } catch (error) {
    console.error('An error occurred:', error);
    return null;
  } finally {
    client.release();
  }
}
async function insertEdge(from_id, to_id, type = 'connection', project) {
  const client = await pool.connect();
  try {
    const query = `
      INSERT INTO "${project}_edges" (from_id, to_id, type)
      VALUES ($1, $2, $3)
      ON CONFLICT (from_id, to_id, type) DO NOTHING
      RETURNING *;
    `;
    const values = [from_id, to_id, type];
    const result = await client.query(query, values);
    return result.rows;
  } catch (error) {
    console.error('An error occurred:', error);
    return null;
  } finally {
    client.release();
  }
}
async function insertDataIntoDatabase(urlString, data, type = 'default', project, page_text = null) {
  const url = new URL(urlString);
  const mainUrl = `${url.protocol}//${url.hostname}`;
  // Insert as a node with type 'domain'
  const nodeId = await insertNode(mainUrl, 'domain', { profileData: { [type]: data } }, project, page_text);
  return nodeId ? [{ id: nodeId }] : null;
}
async function insertPerson(urlString, data, project, page_text = null) {
  // Insert as a node with type 'person'
  const nodeId = await insertNode(urlString, 'person', data, project, page_text);
  return nodeId ? [{ id: nodeId }] : null;
}
async function insertConnection(fromUrl, toUrl, type = 'connection', project) {
  // Get node IDs for from and to
  const fromNode = await getNodeBySId(fromUrl, project);
  const toNode = await getNodeBySId(toUrl, project);
  if (!fromNode || !toNode) return null;
  return await insertEdge(fromNode.id, toNode.id, type, project);
}
async function getNodeBySId(s_id, project) {
  const client = await pool.connect();
  try {
    const query = `SELECT * FROM "${project}_nodes" WHERE s_id = $1`;
    const result = await client.query(query, [s_id]);
    return result.rows[0];
  } finally {
    client.release();
  }
}
// Other ported functions (e.g., calculate_xpath_axis_optimized) can be added similarly for optimized queries
function getClientForModel(model) {
  if (model.startsWith('deepseek')) {
    return new OpenAI({
      baseURL: 'https://api.deepseek.com',
      apiKey: process.env.DEEPSEEK_API_KEY
    });
  }
  return new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
}
function texttoJSON(text) {
try {
return JSON.parse(text);
} catch (e) {
console.error('Invalid JSON:', text);
// Attempt to repair truncated JSON
let repaired = text.trim();
// Parse to build stack and check if in string
let stack = [];
let i = 0;
let inString = false;
let escaped = false;
while (i < repaired.length) {
let char = repaired[i];
if (inString) {
if (escaped) {
escaped = false;
} else if (char === "\\") {
  escaped = true;
} else if (char === '"') {
inString = false;
}
} else {
if (char === '"') {
inString = true;
} else if (char === '{') {
stack.push('{');
} else if (char === '[') {
stack.push('[');
} else if (char === '}') {
if (stack.length && stack[stack.length - 1] === '{') {
stack.pop();
}
} else if (char === ']') {
if (stack.length && stack[stack.length - 1] === '[') {
stack.pop();
}
}
}
i++;
}
// If still in string, close it
if (inString) {
repaired += '"';
}
// Handle trailing commas
repaired = repaired.replace(/,\s*$/, '');
// Close open structures in reverse order
while (stack.length > 0) {
let top = stack.pop();
if (top === '{') {
repaired += '}';
} else if (top === '[') {
repaired += ']';
}
}
try {
const parsed = JSON.parse(repaired);
console.warn('Repaired truncated JSON:', repaired);
return parsed;
} catch (repairError) {
console.error('Repair failed:', repairError);
return { entities: [] }; // Fallback to empty entities array to avoid total loss
}
}
}
async function prompt(systemContent, userContent, model = "deepseek-chat") {
  console.info("[prompt] ↪️ Input", { systemContent, model });
  try {
    const openai = getClientForModel(model);
    const response = await openai.chat.completions.create({
      model,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: systemContent },
        { role: "user", content: userContent },
      ],
    });
    const content = response.choices?.[0]?.message?.content ?? "";
    console.info("[prompt] ✅ Response", content);
    return texttoJSON(content);
  } catch (error) {
    console.error("[prompt] ❌ Error creating response", error);
    throw error;
  }
}
async function processHtmlChunks(systemContent, chunks) {
  const processChunk = async (chunk) => {
    try {
      const response = await prompt(systemContent, chunk);
      return [response]; // Wrap in array for consistency
    } catch (error) {
      if (error.message && error.message.includes('maximum context length')) {
        const mid = Math.floor(chunk.length / 2);
        const half1 = chunk.slice(0, mid);
        const half2 = chunk.slice(mid);
        await new Promise((resolve) => setTimeout(resolve, 500)); // Rate limit delay
        const subResults = await processHtmlChunks(systemContent, [half1, half2]);
        return subResults;
      } else {
        throw error;
      }
    }
  };
  const results = await Promise.all(chunks.map((chunk) => processChunk(chunk)));
  return results.flat(); // Flatten the array of arrays
}
function mergeDetails(allDetails) {
  // Flatten all entities from chunks
  const allEntities = allDetails.flatMap(d => d.entities || []);
  // Group entities by type and key (name for persons, src for images/videos)
  const grouped = {
    person: {},
    image: {},
    video: {}
  };
  allEntities.forEach(entity => {
    const type = entity.type;
    if (!grouped[type]) return;
    let key;
    if (type === 'person') {
      key = entity.name?.toLowerCase() || 'unknown';
    } else if (type === 'image') {
      key = entity.src || entity.base64 || 'unknown';
    } else if (type === 'video') {
      key = entity.src || entity.embed || entity.base64 || 'unknown';
    }
    if (!grouped[type][key]) {
      grouped[type][key] = [];
    }
    grouped[type][key].push(entity);
  });
  // Merge within each group
  const mergedEntities = [];
  Object.keys(grouped).forEach(type => {
    Object.keys(grouped[type]).forEach(key => {
      const group = grouped[type][key];
      if (group.length === 0) return;
      const merged = { type };
      if (type === 'person') {
        merged.name = group[0].name; // Assume name is consistent
        // Merge images: pick the first non-null image
        merged.image = group.find(g => g.image)?.image || null;
      } else if (type === 'image') {
        merged.src = group[0].src;
        merged.base64 = group[0].base64;
        merged.alt = group[0].alt || '';
      } else if (type === 'video') {
        merged.src = group[0].src;
        merged.embed = group[0].embed;
        merged.base64 = group[0].base64;
        merged.alt = group[0].alt || '';
      }
      // Merge details
      const details = {
        publisher: null,
        location: null,
        date: null,
        age: null,
        political_info: null,
        other_details: {}
      };
      const fields = ['publisher', 'location', 'date', 'age', 'political_info'];
      const counts = {};
      group.forEach(g => {
        fields.forEach(f => {
          if (g.details?.[f]) {
            counts[f] = counts[f] || {};
            counts[f][g.details[f]] = (counts[f][g.details[f]] || 0) + 1;
          }
        });
        if (g.details?.other_details) {
          Object.assign(details.other_details, g.details.other_details);
        }
      });
      fields.forEach(f => {
        if (counts[f]) {
          details[f] = Object.keys(counts[f]).reduce((a, b) => counts[f][a] > counts[f][b] ? a : b);
        }
      });
      merged.details = details;
      mergedEntities.push(merged);
    });
  });
  // Assign index numbers to each merged entity
  mergedEntities.forEach((entity, index) => {
    entity.index = index + 1;
  });
  return { entities: mergedEntities };
}
const platformSelectors = {
  'instagram': {
    'follower_button': 'a[href*="/followers/"]',
    'person_item': 'div.x1qnrgzn.x1cek8b2.xb10e19.x19rwo8q.x1lliihq.x193iq5w.xh8yej3',
    'name': 'span[class="_ap3a _aaco _aacw _aacx _aad7 _aade"]',
    'profile_url': 'a[class="x1i10hfl xjbqb8w x1ejq31n x18oe1m7 x1sy0etr xstzfhl x972fbf x10w94by x1qhh985 x14e42zd x9f619 x1ypdohk xt0psk2 xe8uvvx xdj266r x14z9mp xat24cr x1lziwak xexx8yu xyri2b x18d9i69 x1c1uobl x16tdsg8 x1hl2dhg xggy1nq x1a2a7pz notranslate _a6hd"]',
    'next_page': null
  },
  'facebook': {
    'profile': {
      'follower_button': 'a[href*="/followers"]',
      'person_item': 'div[class="x6s0dn4 x1obq294 x5a5i1n xde0f50 x15x8krk x1olyfxc x9f619 x78zum5 x1e56ztr xyamay9 xv54qhq x1l90r2v xf7dkkf x1gefphp"]',
      'name': 'span[class="x193iq5w xeuugli x13faqbe x1vvkbs x10flsy6 x1lliihq x1s928wv xhkezso x1gmr53x x1cpjm7i x1fgarty x1943h6x x1tu3fi x3x7a5m x1lkfr7t x1lbecb7 x1s688f xzsf02u"]',
      'profile_url': 'a[class="x1i10hfl xjbqb8w x1ejq31n x18oe1m7 x1sy0etr xstzfhl x972fbf x10w94by x1qhh985 x14e42zd x9f619 x1ypdohk xt0psk2 xe8uvvx xdj266r x14z9mp xat24cr x1lziwak xexx8yu xyri2b x18d9i69 x1c1uobl x16tdsg8 x1hl2dhg xggy1nq x1a2a7pz x1heor9g xkrqix3 x1sur9pj x1s688f"]',
      'next_page': null
    },
    'group': {
      'follower_button': 'a[href*="/members"]',
      'person_item': 'div.x1obq294.x5a5i1n.xde0f50.x15x8krk.x1lliihq',
      'name': 'a[class="x1i10hfl xjbqb8w x1ejq31n x18oe1m7 x1sy0etr xstzfhl x972fbf x10w94by x1qhh985 x14e42zd x9f619 x1ypdohk xt0psk2 xe8uvvx xdj266r x14z9mp xat24cr x1lziwak xexx8yu xyri2b x18d9i69 x1c1uobl x16tdsg8 x1hl2dhg xggy1nq x1a2a7pz xkrqix3 x1sur9pj xzsf02u x1pd3egz"]',
      'profile_url': 'a[class="x1i10hfl xjbqb8w x1ejq31n x18oe1m7 x1sy0etr xstzfhl x972fbf x10w94by x1qhh985 x14e42zd x9f619 x1ypdohk xt0psk2 xe8uvvx xdj266r x14z9mp xat24cr x1lziwak xexx8yu xyri2b x18d9i69 x1c1uobl x16tdsg8 x1hl2dhg xggy1nq x1a2a7pz xkrqix3 x1sur9pj xzsf02u x1pd3egz"]',
      'next_page': null
    }
  },
  'twitter': {
    'follower_button': 'a[href*="/verified_followers"]',
    'person_item': 'div[class="css-175oi2r r-1adg3ll r-1ny4l3l"]',
    'name': 'div[class="css-1jxf684 r-bcqeeo r-1ttztb7 r-qvutc0 r-poiln3"]',
    'profile_url': 'a[class="css-175oi2r r-1wbh5a2 r-dnmrzs r-1ny4l3l r-1loqt21"]',
    'next_page': null
  },
  'tiktok': {
    'follower_button': 'span[data-e2e="followers"]',
    'person_item': 'div[class="css-14xr620-DivUserContainer ex4st9p0"]',
    'name': 'span[class="css-k0d282-SpanNickname ex4st9p6"]',
    'profile_url': 'a[class="css-7fu252-StyledUserInfoLink ex4st9p3 link-a11y-focus"]',
    'next_page': null
  },
  'gettr': {
    'follower_button': 'a[href*="/followers"]',
    'person_item': 'div[class="jss1308"]',
    'name': 'p[class="jss1317 jss1347 followDisplayName"]',
    'profile_url': 'div > a',
    'next_page': null
  }
};
function getPlatform(sUrl) {
  const host = new URL(sUrl).hostname.toLowerCase();
  if (host.includes('instagram')) return 'instagram';
  if (host.includes('facebook') || host.includes('fb')) return 'facebook';
  if (host.includes('twitter') || host.includes('x.com')) return 'twitter';
  if (host.includes('tiktok')) return 'tiktok';
  if (host.includes('gettr')) return 'gettr';
  return null;
}
class Scraper {
  constructor(wss) {
    this.browser = null;
    this.page = null;
    this.lastKnownUrl = null;
    this.wss = wss;
  }
  sendToWS(data) {
    this.wss.clients.forEach(client => {
      if (client.readyState === 1) {
        client.send(JSON.stringify(data));
      }
    });
  }
  async init() {
    this.browser = await puppeteer.launch({ headless: false });
    this.page = await this.browser.newPage();
    await this.page.setUserAgent('Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0');
    await this.page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1, fullPage: true });
  }
  async attemptLoop(checkFn, breakWhen = true, page = this.page) {
    let attempts = 0;
    while (attempts < 50) {
      const result = await checkFn();
      if (result === breakWhen) {
        break;
      }
      this.sendToWS({ type: 'console_log', message: 'user interaction is needed' });
      const waitResult = await new Promise(async (resolve) => {
        const checkInterval = setInterval(() => {
          if (breakSignal) {
            breakSignal = false;
            clearInterval(checkInterval);
            resolve('break');
          }
        }, 500);
        try {
          await page.waitForNavigation({ waitUntil: 'networkidle2', timeout: 10000 });
          clearInterval(checkInterval);
          resolve('navigation');
        } catch (e) {
          clearInterval(checkInterval);
          resolve('timeout');
        }
      });
      if (waitResult === 'break') {
        this.sendToWS({ type: 'console_log', message: 'breaking attempt loop' });
        break;
      }
      attempts++;
    }
    if (attempts >= 50) {
      throw new Error("Failed after 50 attempts to check page");
    }
    this.sendToWS({ type: 'console_log', message: 'continue to scraping' });
  }
  async scrollWhilePageChanges(page = this.page) {
    try {
      const scrollStep = 5500; // Define the scroll step size (in pixels)
      const scrollDelay = 1000; // Define the delay between scrolls (in milliseconds)
      const checkInterval = 1500; // Define the interval for checking changes (in milliseconds)
      let previousContent = await page.content();
      let count = 0;
      while (count <= 100) {
        await page.evaluate((step) => window.scrollBy(0, step), scrollStep);
        await new Promise(resolve => setTimeout(resolve, scrollDelay));
        let currentContent = await page.content();
        if (currentContent !== previousContent) {
          previousContent = currentContent;
        } else {
            break;
        }
        previousContent = currentContent;
        count++;
      }
    } catch (error) {
      console.error('Error during page scroll:', error);
    }
  }
  async copyAllTextContent(page = this.page) {
    try {
        const allText = await page.evaluate(() => {
        return document.body.innerText;
        });
        return allText;
    } catch (error) {
        console.log(error);
        return null;
    }
  }
  async searchAndGetLinks(query) {
    let links = [];
    try {
      await this.page.goto(`https://www.google.com`, { waitUntil: 'networkidle0' });
      await this.attemptLoop(async () => await this.check(this.page), false, this.page);
      const textareaSelector = 'textarea[name="q"]';
      const inputSelector = 'input[name="q"]';
      let searchBox;
      try {
        await this.page.waitForSelector(textareaSelector, { timeout: 2000 });
        searchBox = textareaSelector;
      } catch {
        await this.page.waitForSelector(inputSelector, { timeout: 2000 });
        searchBox = inputSelector;
      }
      await this.page.click(searchBox);
      await this.page.type(searchBox, query);
      await this.page.keyboard.press('Enter');
      await this.page.waitForNavigation({ waitUntil: 'networkidle0' });
      await this.attemptLoop(async () => await this.check(this.page), false, this.page);
      while (true) {
        await this.scrollWhilePageChanges(this.page);
        const newLinks = await this.page.evaluate(() => {
          return Array.from(document.querySelectorAll('a.zReHs'))
            .map(link => link.href)
            .filter(href => href.startsWith('http') && !href.includes('google.com'));
        });
        links = [...new Set([...links, ...newLinks])];
        const nextButton = await this.page.$('#pnnext');
        if (!nextButton) break;
        await nextButton.click();
        await this.page.waitForNavigation({ waitUntil: 'networkidle0' });
      }
    } catch (error) {
      console.error('Error during search and link collection:', error);
    }
    return links;
  }
  async searchAndGetImageLinks(query) {
    let imageLinks = [];
    try {
      await this.page.goto(`https://www.google.com`, { waitUntil: 'networkidle0' });
      await this.attemptLoop(async () => await this.check(this.page), false, this.page);
      const textareaSelector = 'textarea[name="q"]';
      const inputSelector = 'input[name="q"]';
      let searchBox;
      try {
        await this.page.waitForSelector(textareaSelector, { timeout: 2000 });
        searchBox = textareaSelector;
      } catch {
        await this.page.waitForSelector(inputSelector, { timeout: 2000 });
        searchBox = inputSelector;
      }
      await this.page.click(searchBox);
      await this.page.type(searchBox, query);
      await this.page.keyboard.press('Enter');
      await this.page.waitForNavigation({ waitUntil: 'networkidle0' });
      await this.attemptLoop(async () => await this.check(this.page), false, this.page);
      const imageTab = await this.getContentTranslation("Images Tab", this.page);
      await this.page.evaluate((textContent) => {
        const link = Array.from(document.querySelectorAll('a')).find(a => a.textContent.includes(textContent));
        if (link) {
          link.click();
        }
      }, imageTab);
      await this.page.waitForNavigation({ waitUntil: 'networkidle0' });
      while (true) {
        const initialCount = imageLinks.length;
        const newImageLinks = await this.page.evaluate(() => {
          return Array.from(document.querySelectorAll('a.EZAeBe'))
            .map(link => link.href)
            .filter(href => href.startsWith('http') && !href.includes('google.com'));
        });
        imageLinks = [...new Set([...imageLinks, ...newImageLinks])];
        await this.scrollWhilePageChanges(this.page);
        await new Promise(resolve => setTimeout(resolve, 3000));
        if (imageLinks.length === initialCount) break;
      }
    } catch (error) {
      console.error('Error during image search and collection:', error);
    }
    return imageLinks;
  }
  async close() {
    if (this.browser) await this.browser.close();
  }
  async filterHTMLContent(page = this.page) {
    const newPage = await page.browser().newPage();
    await newPage.setContent(await page.content());
    const modifiedContent = await newPage.evaluate(() => {
      const uniqueHrefs = new Set();
      const uniqueSrcs = new Set();
      function cleanAttributes(element) {
        const allowedAttributes = ['href', 'src', 'class', 'id'];
        for (const attr of Array.from(element.attributes)) {
          const attrName = attr.name;
          const attrValue = attr.value;
          if (attrName === 'href') {
            if (uniqueHrefs.has(attrValue)) {
              element.removeAttribute(attrName);
            } else {
              uniqueHrefs.add(attrValue);
            }
          } else if (attrName === 'src') {
            if (uniqueSrcs.has(attrValue)) {
              element.removeAttribute(attrName);
            } else {
              uniqueSrcs.add(attrValue);
            }
          } else if (!allowedAttributes.includes(attrName)) {
            element.removeAttribute(attrName);
          }
        }
        Array.from(element.children).forEach(cleanAttributes);
      }
      cleanAttributes(document.body);
      document.querySelectorAll(['style', 'script', 'link', 'meta', 'path', 'noscript', 'source']).forEach(el => el.remove());
      document.querySelectorAll([
        'meta', 'path', 'noscript', 'script', 'source', 'defs', 'symbol',
        'use', 'image', 'switch', 'desc', 'metadata', 'rect', 'circle', 'ellipse',
        'line', 'polyline', 'polygon', 'text', 'tspan', 'tref', 'textPath',
        'altGlyph', 'altGlyphDef', 'altGlyphItem', 'glyphRef', 'textPath',
        'linearGradient', 'radialGradient', 'stop', 'feBlend', 'feColorMatrix',
        'feComponentTransfer', 'feComposite', 'feConvolveMatrix', 'feDiffuseLighting',
        'feDisplacementMap', 'feDistantLight', 'feDropShadow', 'feFlood', 'feFuncA',
        'feFuncB', 'feFuncG', 'feFuncR', 'feGaussianBlur', 'feImage', 'feMerge',
        'feMergeNode', 'feMorphology', 'feOffset', 'fePointLight', 'feSpecularLighting',
        'feSpotLight', 'feTile', 'feTurbulence', 'animate', 'animateColor',
        'animateMotion', 'animateTransform', 'mpath', 'set', 'clipPath',
        'color-profile', 'cursor', 'filter', 'foreignObject', 'hatch', 'hatchpath',
        'marker', 'mask', 'pattern', 'view'
      ]).forEach(el => {
        while (el.firstChild) {
          el.parentNode.insertBefore(el.firstChild, el);
        }
        el.remove();
      });
      return document.documentElement.outerHTML;
    });
    await newPage.close();
    return modifiedContent;
  };
  async check(page = this.page) {
    const currentPageContent = await this.filterHTMLContent(page);
    const systemContent = `You are a helpful assistant that decides if the given HTML requires a user interaction like an account login (but only if the login is required and not optional), or a captcha to solve or an accept cookies or similar to continue. An account login is considered needed only if no other content is visible and not if login is optional.
      1. You get the complete HTML Content of a Website.
      2. You identify the best fitting HTML Element in the HTML Content, that indicates an user interaction like, account login (but only if the login is from the main page and required, not optional) or captcha or cookies accept button (when the cookies banner is hidden then also return false) requirement and rank your answer from 1 to 10 (10 is best).
      3. You return true if an account login, captcha, or cookies is need, else return false.
      4. userInteraction is not needed if its indicating an optional action.
      Provide the result in a clear JSON format.
      If the element of interest exists, you return:
      {
          "userInteraction": "true/false",
          "type": "type of user interaction needed: account, captcha, cookies, ...",
          "rating": "rating number",
          "explanation": "why you chose this text with outerHTML"
        }`;
    const outerHTMLs = await processHtmlChunks(systemContent, [currentPageContent]);
    const processed = outerHTMLs.map(r => ({ rating: r.rating || 0, userInteraction: r.userInteraction || "false" }));
    const bestChunk = processed.reduce((prev, current) => (parseInt(current.rating) > parseInt(prev.rating)) ? current : prev, processed[0] || { rating: 0, userInteraction: "false" });
    console.warn(bestChunk);
    if (bestChunk.userInteraction === "true") {
      console.warn("user interaction true")
      return true;
    } else {
      console.warn("user interaction false")
      return false;
    }
  }
  async getContentTranslation(query, page = this.page) {
    try {
      const currentPageContent = await this.filterHTMLContent(page);
      const systemContent = `You are a helpful assistant that analyzes HTML content to find the word or phrase in the page's language that best corresponds to the English term "${query}" (e.g., if the page is in German, find "Bilder" for "Images").
          **Instructions:**
          1. Analyze the provided HTML chunk from a website.
          2. Detect the primary language of the content if possible.
          3. Identify the best-fitting text content (e.g., link text, button label, or heading) that semantically matches "${query}".
          4. Rate your confidence in the match from 1 to 10 (10 being a perfect fit, e.g., exact translation in context).
          5. If no match, return rating 0 and word as empty string.
          6. Include an explanation with the outerHTML of the chosen element.
          **Output Format (JSON only):**
          {
            "word": "the matched word or phrase",
            "rating": number,
            "explanation": "why you chose this, including outerHTML"
          }
          `;
      const evaluations = await processHtmlChunks(systemContent, [currentPageContent]);
      if (evaluations.length === 0) {
        throw new Error('No content chunks available');
      }
      const bestEvaluation = evaluations.reduce((prev, current) => {
        return (parseInt(current.rating || 0) > parseInt(prev.rating || 0)) ? current : prev;
      }, evaluations[0]);
      console.log('Best evaluation:', bestEvaluation);
      if ((bestEvaluation.rating || 0) < 5) {
        console.warn('Low confidence in translation; consider manual review');
      }
      return bestEvaluation.word || '';
    } catch (error) {
      console.error('Error in getContentTranslation:', error);
      return '';
    }
  }
  async getDetails(query, page = this.page) {
    const currentPageContent = await this.filterHTMLContent(page);
    const url = await page.url();
    const systemContent = `You are a helpful assistant that scrapes detailed information from a given HTML content. Focus on extracting data relevant to the topic: "${query}". Your task is to extract persons, images, and videos, associating images to persons where possible (e.g., if an image is a profile picture or nearby in the HTML). Each extracted entry must be either a person (with optional image and details), a standalone image (with details), or a standalone video (with details). Do not extract other types of data.
        **Instructions:**
        1. You receive a chunk of HTML content from a website.
        2. Identify and extract entities: persons, images, videos.
        3. For persons: Extract name, attach an image if it appears to be associated (e.g., profile image), and details.
        4. For standalone images (not associated with a person): Extract src/base64, alt, and any associated details.
        5. For videos: Extract src/embed/base64, alt/description, and any associated details.
        6. Details for each entity include (if available and relevant to that entity):
           - Publisher (e.g., author, organization)
           - Nationality (e.g., race, geographic location)
           - Language (e.g., related language)
           - Gender (e.g., gender of person)
           - Location (e.g., geo data, geographic location)
           - Date (e.g., publication, event date or any type of date related)
           - Time (e.g., any kind of related time stamp)
           - Age (e.g., age of person, as an single number)
           - Profession (e.g., somebodys work)
           - Political information (e.g., affiliations, stances)
           - Group information (e.g., affiliations to any groups)
           - Other relevant details (e.g., categories, tags, metadata, but not long text)
        7. If a detail is not found for an entity, remove it from the list.
        8. For images/videos in entities, use objects with either:
           - {"src": "URL", "alt": "text"}
           - {"base64": "data", "alt": "text"}
        9. For videos, additionally allow {"embed": "URL", "alt": "text"}.
        10. make sure to always return an full src to this "${url}"
        11. Return an array of entities in JSON format.
        12. Do not return empty objects or null values.
        **Output Format:**
        {
          "entities": [
            {
              "type": "person",
              "name": string,
              "image": {"src": string, "alt": string} | {"base64": string, "alt": string},
              "details": {
                "publisher": string,
                "nationality": string,
                "language": string,
                "gender": string,
                "location": string,
                "date": string,
                "time": string,
                "age": string,
                "profession": string,
                "political_info": string,
                "group": string,
                "other_details": { [key: string]: any }
              }
            },
            {
              "type": "image",
              "src": string,
              "base64": string,
              "details": { ... same as above }
            },
            {
              "type": "video",
              "src": string,
              "embed": string,
              "base64": string,
              "details": { ... same as above }
            }
          ]
        }
        `;
    const detailsList = await processHtmlChunks(systemContent, [currentPageContent]);
    const allDetails = detailsList;
    console.log("Extracted details:", allDetails);
    return {
      details: allDetails,
    };
  }
  async checkPageType(page = this.page) {
    const currentPageContent = await this.filterHTMLContent(page);
    const systemContent = `You are a helpful assistant that analyzes HTML content to determine the type of webpage.
        **Instructions:**
        1. You receive a chunk of HTML content from a website.
        2. Classify the social media page type type as one of the following: {"type": "profile" | "group" | "content" | "personlist" | "other"}.
        3. group is another other page containing a link to a personlist like an event or group related.
        3. Rank your answer from 1 to 10 (10 is best).
        4. Return the result in a clear JSON format with the page type.
        **Output Format:**
        {
            "type": "profile" | "group" | "content" | "personlist" | "other",
            "rating": "rating number"
        }
        `;
    const outerHTMLs = await processHtmlChunks(systemContent, [currentPageContent]);
    const processed = outerHTMLs.map(r => ({ rating: r.rating || 0, type: r.type || "other" }));
    const bestChunk = processed.reduce((prev, current) => (parseInt(current.rating) > parseInt(prev.rating)) ? current : prev, processed[0] || { rating: 0, type: "other" });
    console.warn("Best chunk:", bestChunk);
    return { type: bestChunk.type };
  }
  async checkIfPersonList(page = this.page) {
    const currentPageContent = await this.filterHTMLContent(page);
    const systemContent = `You are a helpful assistant that analyzes HTML content to determine if the page is a list of persons on social media.
        **Instructions:**
        1. You receive a chunk of HTML content from a website.
        2. Determine if it looks like a list of persons (e.g., friends, followers, connections list), but only if its a typical list with only profile image, name, profile url, etc, not if its content page where different people have posted something.
        3. Rank your confidence from 1 to 10 (10 is best).
        4. Return true if it's a person list, else false.
        5. Return the result in a clear JSON format.
        **Output Format:**
        {
            "isPersonList": "true/false",
            "rating": "rating number",
            "explanation": "brief explanation"
        }
        `;
    const results = await processHtmlChunks(systemContent, [currentPageContent]);
    const best = results.reduce((prev, current) => {
      return (parseInt(current.rating || 0) > parseInt(prev.rating || 0)) ? current : prev;
    }, results[0] || { rating: 0, isPersonList: "false" });
    console.warn("Person list check:", best);
    return best.isPersonList === "true";
  }
  async checkIfProfilePage(page = this.page) {
    const currentPageContent = await this.filterHTMLContent(page);
    const systemContent = `You are a helpful assistant that analyzes HTML content to determine if the page is a profile page of a persons on social media.
        **Instructions:**
        1. You receive a chunk of HTML content from a website.
        2. Determine if it looks like a profile page of persons (e.g., friends, followers, connections list).
        3. Rank your confidence from 1 to 10 (10 is best).
        4. Return true if it's a profile page, else false.
        5. Return the result in a clear JSON format.
        **Output Format:**
        {
            "isProfilePage": "true/false",
            "rating": "rating number",
            "explanation": "brief explanation"
        }
        `;
    const results = await processHtmlChunks(systemContent, [currentPageContent]);
    const best = results.reduce((prev, current) => {
      return (parseInt(current.rating || 0) > parseInt(prev.rating || 0)) ? current : prev;
    }, results[0] || { rating: 0, isProfilePage: "false" });
    console.warn("Profile page check:", best);
    return best.isProfilePage === "true";
  }
  async getNextPageButtonSelector(page = this.page) {
    const currentPageContent = await this.filterHTMLContent(page);
    const systemContent = `You are a helpful assistant that identifies a next page button of the person list in HTML content.
        **Instructions:**
        1. You receive a chunk of HTML content from a website.
        2. Look for an element that indicates a "next page" button or link (e.g., "Next", "Load more", pagination arrow).
        3. If found, return the best CSS selector for it.
        4. Rank your confidence from 1 to 10 (10 is best).
        5. If no next button, return null for selector.
        6. Return the result in a clear JSON format.
        **JSON Output Format:**
        {
            "hasNextButton": "true/false",
            "selector": "CSS selector or null",
            "rating": "rating number",
            "explanation": "brief explanation"
        }
        `;
    const results = await processHtmlChunks(systemContent, [currentPageContent]);
    const best = results.reduce((prev, current) => {
      return (parseInt(current.rating || 0) > parseInt(prev.rating || 0)) ? current : prev;
    }, results[0] || { rating: 0, hasNextButton: "false", selector: null });
    console.warn("Next page button check:", best);
    return { hasNextButton: best.hasNextButton === "true", selector: best.selector };
  }
  async getFriendsButtonSelector(page = this.page) {
    const currentPageContent = await this.filterHTMLContent(page);
    const systemContent = `You are a helpful assistant that identifies a friends/followers button in a social media profile HTML.
        **Instructions:**
        1. You receive a chunk of HTML content from a social media profile.
        2. Look for a button or link to view friends, followers, members or person lists (e.g., "Friends", "Followers", "Members").
        3. Return the best CSS selector for it.
        4. Rank your confidence from 1 to 10 (10 is best).
        5. If not found, return null.
        6. Return the result in a clear JSON format.
        **JSON Output Format:**
        {
            "selector": "CSS selector or null",
            "rating": "rating number",
            "explanation": "brief explanation"
        }
        `;
    const results = await processHtmlChunks(systemContent, [currentPageContent]);
    const best = results.reduce((prev, current) => {
      return (parseInt(current.rating || 0) > parseInt(prev.rating || 0)) ? current : prev;
    }, results[0] || { rating: 0, selector: null });
    console.warn("Friends button:", best);
    return best.selector;
  }
  async extractPersonsFromList(sUrl, selectors = null, page = this.page) {
    if (selectors) {
      const persons = [];
      try {
        await page.waitForSelector(selectors.person_item, { timeout: 5000 });
        const items = await page.$$(selectors.person_item);
        for (const item of items) {
          const nameEl = await item.$(selectors.name);
          const name = await nameEl?.evaluate(el => el.textContent?.trim());
          const linkEl = await item.$(selectors.profile_url);
          const profileLink = await linkEl?.evaluate(el => el.getAttribute('href'));
          let profileurl = profileLink;
          if (profileLink && !profileLink.startsWith('http')) {
            profileurl = new URL(profileLink, sUrl).toString();
          }
          if (name && profileurl) {
            persons.push({ name, profileurl });
          }
        }
      } catch (e) {
        console.error('Error extracting with selectors:', e);
      }
      const uniquePersons = Array.from(new Set(persons.map(p => p.profileurl))).map(url => persons.find(p => p.profileurl === url));
      console.log(uniquePersons);
      return uniquePersons || [];
    } else {
      const currentPageContent = await this.filterHTMLContent(page);
      const systemContent = `You are a helpful assistant that extracts person items from a social media list in HTML.
          **Instructions:**
          1. You receive a chunk of HTML content from a person list page.
          2. Identify repeating person items and extract for each: name, profileurl, image.
          3. Create a complete profile_url based on "${sUrl}"
          4. Return an array of objects: [{"name": string, "profileurl": string}].
          **JSON Output Format:**
          {
              "persons": [{"name": "...", "profileurl": "..."}]
          }
          `;
      const personsLists = await processHtmlChunks(systemContent, [currentPageContent]);
      const allPersons = personsLists.flatMap(list => list.persons || []);
      const uniquePersons = Array.from(new Set(allPersons.map(p => p.profileurl))).map(url => allPersons.find(p => p.profileurl === url));
      console.log(uniquePersons);
      return uniquePersons;
    }
  }
  async processNonSocialLinks(nonSocialLinks, query, project, processed, processedFile) {
    const concurrency = 10;
    const browser = this.browser;
    const tasks = nonSocialLinks.map(url => async () => {
      if (processed.nonSocial.includes(url)) return;
      try {
        const page = await browser.newPage();
        await page.setUserAgent('Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0');
        await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });
        this.sendToWS({ type: 'console_log', message: `Scraping non-social URL: ${url}` });
        await page.goto(url, { waitUntil: 'domcontentloaded' });
        await this.scrollWhilePageChanges(page);
        const pageText = await this.copyAllTextContent(page);
        const { details } = await this.getDetails(query, page);
        let extracted = mergeDetails(details);
        await insertDataIntoDatabase(url, extracted, 'nonSocial', project, pageText);
        processed.nonSocial.push(url);
        fs.writeFileSync(processedFile, JSON.stringify(processed, null, 2));
        await page.close();
      } catch (e) {
        console.error(`Error processing non-social ${url}:`, e);
      }
    });
    const queue = [...tasks];
    while (queue.length > 0) {
      const batch = queue.splice(0, concurrency);
      await Promise.all(batch.map(task => task()));
    }
  }
  async processSocialLinks(socialLinks, query, depth, max, project, processed, processedFile) {
    for (const slink of socialLinks) {
      if (processed.social.includes(slink)) continue;
      await this.processSocialUrl(slink, query, depth, max, project, new Set(processed.social), null, processed, processedFile, this.page);
    }
  }
  async processSocialUrl(sUrl, query, currDepth, max, project, visited = new Set(), forcedType = null, processed, processedFile, page = this.page) {
    if (currDepth <= 0 || visited.has(sUrl) || processed.social.includes(sUrl)) return;
    visited.add(sUrl);
    processed.social.push(sUrl);
    fs.writeFileSync(processedFile, JSON.stringify(processed, null, 2));
    try {
      this.sendToWS({ type: 'console_log', message: `Scraping social URL: ${sUrl}` });
      await page.goto(sUrl, { waitUntil: 'networkidle0' });
      let type;
      if (forcedType) {
        type = forcedType;
      } else {
        await this.attemptLoop(async () => await this.check(page), false, page);
        ({ type } = await this.checkPageType(page));
      }
      try {
          await page.evaluate((textContent) => {
            const link = Array.from(document.querySelectorAll('a')).find(a => a.textContent.includes(textContent));
            if (link) {
              link.click();
            }
          }, "Profil ansehen");
    } catch {}

      if (type === 'other') return;
      let pageText;
      if (type === 'content') {
        await this.attemptLoop(async () => await this.checkIfProfilePage(page), true, page);
        type = 'profile';
        sUrl = await this.page.url();
      }
      if (type === 'profile' || type === 'group') {
        let initialContent = await page.content();
        let previousText = initialContent;
        let allContent = previousText;
        let count = 0;
        const maxCount = 30;
        while (count < maxCount) {
            await page.evaluate(() => { window.scrollBy(0, document.body.scrollHeight); });
            await new Promise(resolve => setTimeout(resolve, 500));
            let currentText = await page.content();
            let newText;
            if (currentText.startsWith(previousText)) {
                newText = currentText.slice(previousText.length);
            } else {
                newText = currentText; // Fallback: append full if not a clean append
            }
            allContent += newText;
            previousText = currentText;
            count++;
        }
    const { details } = await this.getDetails(query, page);
    let extracted = mergeDetails(details);
    await insertDataIntoDatabase(sUrl, extracted, 'socialPage', project, allContent);
    await insertPerson(sUrl, { ...extracted, profileurl: sUrl }, project, allContent);
      }

      const platform = getPlatform(sUrl);
      let selectors = null;
      if (platform) {
        if (platform === 'facebook' && (type === 'profile' || type === 'group')) {
          selectors = platformSelectors[platform][type];
        } else {
          selectors = platformSelectors[platform];
        }
      }
            await page.evaluate(() => window.scrollTo(0, 0));

      if (selectors) {
        try {
          await page.waitForSelector(selectors.follower_button, { timeout: 5000 });
          await page.click(selectors.follower_button);
        } catch (e) {
          console.log('Fallback to translation for follower tab');
          const followerTab = await this.getContentTranslation("Follower Tab", page);
          await page.evaluate((textContent) => {
    const link = Array.from(document.querySelectorAll('a')).find(a => a.textContent.trim() === textContent.trim());
    if (link) {
      link.click();
    }
}, followerTab);
        }
      } else {
        const followerTab = await this.getContentTranslation("Follower Tab", page);
       await page.evaluate((textContent) => {
    const link = Array.from(document.querySelectorAll('a')).find(a => a.textContent.trim() === textContent.trim());
    if (link) {
      link.click();
    }
}, followerTab);
      }
      //await this.attemptLoop(async () => await this.checkIfPersonList(page), true, page);
      // Track extracted profile URLs to avoid duplicates
      const processedProfiles = new Set();
      // Scroll or page the list
      let hasMore = true;
      let prevContent = await page.content();
      while (hasMore) {
        // Extract persons from the current page
        const persons = await this.extractPersonsFromList(sUrl, selectors, page);
        // Process new persons
        for (const p of persons) {
          console.log(p.profileurl);
          if (p.profileurl) {
            if (!processedProfiles.has(p.profileurl)) {
              processedProfiles.add(p.profileurl);
              await insertPerson(p.profileurl, p, project);
              await insertConnection(sUrl, p.profileurl, 'connection', project);
              await insertConnection(p.profileurl, sUrl, 'connection', project);
            }
          }
        }
        if (processedProfiles.size >= max) {
          hasMore = false;
        }
        // Check for next page or scroll
        let hasNextButton = false;
        let nextSelector = null;
        if (selectors && selectors.next_page) {
          nextSelector = selectors.next_page;
          hasNextButton = !!await page.$(nextSelector);
        } else if (selectors && selectors.next_page === null) {
        } else {
          const { hasNextButton: llmHasNext, selector: llmSelector } = await this.getNextPageButtonSelector(page);
          hasNextButton = llmHasNext;
          nextSelector = llmSelector;
        }
        if (hasNextButton && nextSelector) {
          const nextBtn = await page.$(nextSelector);
          if (nextBtn) {
            await nextBtn.click();
            await new Promise((resolve) => setTimeout(resolve, 3000));
            await page.waitForNetworkIdle({ idleTime: 1000 });
          } else {
            hasMore = false;
          }
        } else {
            await page.bringToFront();
          // Move mouse to the middle of the page
            await page.mouse.move(1920 / 2, 1080 / 2);
            // Get the total scroll height
            const scrollHeight = await page.evaluate(() => document.body.scrollHeight);
            // Scroll in steps using mouse wheel
            const step = 500;
            for (let pos = 0; pos < scrollHeight; pos += step) {
            await page.mouse.wheel({ deltaY: step });
          await new Promise((resolve) => setTimeout(resolve, 500));
            }
          //await this.scrollWhilePageChanges(page);
          const currContent = await page.content();
          if (currContent === prevContent) {
            hasMore = false;
          } else {
            prevContent = currContent;
          }
        }
      }


     /* for (const profileUrl of processedProfiles) {
        await this.processSocialUrl(profileUrl, profileUrl, currDepth - 1, project, visited, 'profile', processed, processedFile, page);
      }*/

      const limit = pLimit(10);
        await Promise.all(
        Array.from(processedProfiles).map(profileUrl =>
            limit(async () => {
            const newPage = await this.browser.newPage();
            try {
                await newPage.setViewport({ width: 1920, height: 1080 });
                await this.processSocialUrl(
                profileUrl,
                profileUrl,
                currDepth - 1,
                project,
                visited,
                'profile',
                processed,
                processedFile,
                newPage
                );
            } catch (err) {
                console.error(`Error in parallel processing of ${profileUrl}:`, err);
            } finally {
                await newPage.close();
            }
            })
        )
        );
    } catch (e) {
      console.error(`Error processing social ${sUrl}:`, e);
    }
  }
}
async function loadPersonsData(project) {
  const client = await pool.connect();
  try {
    const nodesQuery = `SELECT id, s_id, type, content FROM "${project}_nodes"`;
    const nodesResult = await client.query(nodesQuery);
    console.log('Total nodes retrieved:', nodesResult.rows.length);

    let allEntityItems = [];

    nodesResult.rows.forEach(row => {
      const content = row.content || {};
      const directEntities = content.entities || [];
      const profileEntities = Object.values(content.profileData || {}).flatMap(pd => pd.entities || []);
      const allEntities = [...directEntities, ...profileEntities].filter(
        e => e && ['person', 'image', 'video'].includes(e.type)
      );

      allEntities.forEach((e, index) => {
        let attributes = e.details || {};
        let images = [];
        let videos = [];

        if (e.type === 'person') {
          const imgSrc = e.image
            ? (e.image.src || (e.image.base64 ? `data:image/png;base64,${e.image.base64}` : null))
            : null;
          if (imgSrc) images = [imgSrc];
        } else if (e.type === 'image') {
          const imgSrc = e.src || (e.base64 ? `data:image/png;base64,${e.base64}` : null);
          if (imgSrc) {
            images = [imgSrc];
            attributes = { ...e.details, alt: e.alt || 'Image' };
          }
        } else if (e.type === 'video') {
          const vidSrc = e.src || e.embed || (e.base64 ? `data:video/mp4;base64,${e.base64}` : null);
          if (vidSrc) {
            videos = [vidSrc];
            attributes = { ...e.details, alt: e.alt || 'Video' };
          }
        }

        // Create a minimal entity object for transmission
        let full_entity = { ...e };
        // Truncate base64 data for the full_entity field
        if (full_entity.image && full_entity.image.base64) {
          full_entity.image.base64 = '[base64 truncated]';
        }
        if (full_entity.base64) {
          full_entity.base64 = '[base64 truncated]';
        }

        const entityItem = {
          node_id: row.id,
          entity_index: e.index !== undefined ? e.index : index,
          type: e.type,
          name: e.name || e.alt || e.type,
          attributes,
          images: images.map(img => {
            // Limit base64 strings in images array
            if (img && img.startsWith('data:') && img.length > 100000) {
              return img.substring(0, 100000) + '...';
            }
            return img;
          }),
          videos: videos.map(vid => {
            // Limit base64 strings in videos array
            if (vid && vid.startsWith('data:') && vid.length > 100000) {
              return vid.substring(0, 100000) + '...';
            }
            return vid;
          }),
          full_entity
        };

        allEntityItems.push(entityItem);
      });
    });

    console.log(`Total entities to send: ${allEntityItems.length}`);

    // Batch sending configuration
    const BATCH_SIZE = 50; // Send 50 entities at a time
    const MAX_BATCH_STRING_SIZE = 5 * 1024 * 1024; // 5MB max per batch

    // Function to estimate JSON string size
    function estimateSize(obj) {
      try {
        return JSON.stringify(obj).length;
      } catch (e) {
        return MAX_BATCH_STRING_SIZE + 1; // Force new batch if can't stringify
      }
    }

    // Send initial message indicating batch transmission
    wss.clients.forEach(client => {
      if (client.readyState === 1) {
        client.send(JSON.stringify({
          type: 'batch_start',
          totalBatches: Math.ceil(allEntityItems.length / BATCH_SIZE),
          totalItems: allEntityItems.length
        }));
      }
    });

    // Send data in batches
    let batchNumber = 0;
    let currentBatch = [];
    let currentBatchSize = 0;

    for (let i = 0; i < allEntityItems.length; i++) {
      const item = allEntityItems[i];
      const itemSize = estimateSize(item);

      // Check if adding this item would exceed batch limits
      if (currentBatch.length >= BATCH_SIZE ||
          (currentBatchSize + itemSize > MAX_BATCH_STRING_SIZE && currentBatch.length > 0)) {
        // Send current batch
        const message = {
          type: 'data_update_batch',
          batchNumber: batchNumber++,
          persons: currentBatch,
          isLastBatch: false
        };

        try {
          const messageStr = JSON.stringify(message);
          wss.clients.forEach(client => {
            if (client.readyState === 1) {
              client.send(messageStr);
            }
          });
          console.log(`Sent batch ${batchNumber} with ${currentBatch.length} items`);
        } catch (err) {
          console.error(`Error sending batch ${batchNumber}:`, err);
          // If batch still too large, send items individually
          currentBatch.forEach((singleItem, idx) => {
            try {
              const singleMessage = {
                type: 'data_update_batch',
                batchNumber: batchNumber++,
                persons: [singleItem],
                isLastBatch: false
              };
              wss.clients.forEach(client => {
                if (client.readyState === 1) {
                  client.send(JSON.stringify(singleMessage));
                }
              });
            } catch (singleErr) {
              console.error(`Error sending individual item ${idx}:`, singleErr);
            }
          });
        }

        // Reset batch
        currentBatch = [];
        currentBatchSize = 0;

        // Add small delay to prevent overwhelming the client
        await new Promise(resolve => setTimeout(resolve, 50));
      }

      // Add item to current batch
      currentBatch.push(item);
      currentBatchSize += itemSize;
    }

    // Send remaining items in the last batch
    if (currentBatch.length > 0) {
      const message = {
        type: 'data_update_batch',
        batchNumber: batchNumber++,
        persons: currentBatch,
        isLastBatch: true
      };

      try {
        const messageStr = JSON.stringify(message);
        wss.clients.forEach(client => {
          if (client.readyState === 1) {
            client.send(messageStr);
          }
        });
        console.log(`Sent final batch ${batchNumber} with ${currentBatch.length} items`);
      } catch (err) {
        console.error(`Error sending final batch:`, err);
        // Send items individually as fallback
        currentBatch.forEach((singleItem, idx) => {
          try {
            const singleMessage = {
              type: 'data_update_batch',
              batchNumber: batchNumber++,
              persons: [singleItem],
              isLastBatch: idx === currentBatch.length - 1
            };
            wss.clients.forEach(client => {
              if (client.readyState === 1) {
                client.send(JSON.stringify(singleMessage));
              }
            });
          } catch (singleErr) {
            console.error(`Error sending individual item ${idx}:`, singleErr);
          }
        });
      }
    }

    // Send batch complete message
    wss.clients.forEach(client => {
      if (client.readyState === 1) {
        client.send(JSON.stringify({
          type: 'batch_complete',
          totalBatches: batchNumber,
          totalItems: allEntityItems.length
        }));
      }
    });

    console.log(`Batch transmission complete: ${batchNumber} batches sent`);

  } catch (err) {
    console.error('Error loading persons data:', err);
    // Send error message to clients
    wss.clients.forEach(client => {
      if (client.readyState === 1) {
        client.send(JSON.stringify({
          type: 'error',
          message: 'Error loading persons data',
          error: err.message
        }));
      }
    });
  } finally {
    client.release();
  }
}

async function loadGraphData(project) {
  const client = await pool.connect();
  try {
    // Query to get all node content for graph
    const nodesQuery = `SELECT id, s_id, type, content FROM "${project}_nodes"`;
    const nodesResult = await client.query(nodesQuery);
    const graphNodes = nodesResult.rows.map(row => ({
      id: row.id,
      label: (row.content && row.content.name) || row.s_id.split('/').pop() || row.type,
      group: row.type,
    }));
    const graphEdgesQuery = `SELECT from_id, to_id FROM "${project}_edges"`;
    const gEdgesResult = await client.query(graphEdgesQuery);
    const graphEdges = gEdgesResult.rows.map(row => ({
      from: row.from_id,
      to: row.to_id,
    }));
    wss.clients.forEach(client => {
      if (client.readyState === 1) {
        client.send(JSON.stringify({ type: 'graph_data', nodes: graphNodes, edges: graphEdges }));
      }
    });
  } catch (err) {
    console.error('Error loading graph data:', err);
  } finally {
    client.release();
  }
}
const PORT = process.env.PORT || 4000;
const app = express();
const server = http.createServer(app);
const io = new Server(server);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DOWNLOADS = path.join(__dirname, 'temp');
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ limit: '50mb', extended: true }));
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, 'uploads/'),
  filename: (req, file, cb) => cb(null, Date.now() + '-' + Math.round(Math.random() * 1E9) + path.extname(file.originalname))
});
const upload = multer({ storage, limits: { fileSize: 5 * 1024 * 1024 } });
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'osint-map.html'));
});
app.post('/process-data', upload.single('image'), async (req, res) => {
  try {
    const configText = req.body.config;
    let parsedConfig = JSON.parse(configText);
    const projectName = parsedConfig.query.replace(/\s+/g, '_');
    const newPath = path.join(ROOT_DOWNLOADS, projectName);
    if (!fs.existsSync(newPath)) {
      fs.mkdirSync(newPath, { recursive: true });
    }
    parsedConfig.project = projectName;
    const configParam = encodeURIComponent(JSON.stringify(parsedConfig));
    res.redirect(`/scraper?config=${configParam}`);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Error processing data' });
  }
});
let breakSignal = false;
const wss = new WebSocketServer({ port: 4002 });
wss.on('connection', (ws) => {
  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message);
      if (data.type === 'break_attempt_loop') {
        breakSignal = true;
      } else if (data.type === 'request_data') {
        const project = data.project;
        const projectName = project.replace(/\s+/g, '_');
        if (projectName) {
          loadPersonsData(projectName);
        }
      } else if (data.type === 'request_graph') {
        const project = data.project;
        const projectName = project.replace(/\s+/g, '_');
        if (projectName) {
          loadGraphData(projectName);
        }
      }
    } catch (e) {
      console.error('WebSocket message error:', e);
    }
  });
});
app.get('/scraper', async (req, res) => {
  try {
    const config = req.query.config;
    if (!config) return res.status(400).json({ error: 'Configuration not provided' });
    const parsedConfig = JSON.parse(decodeURIComponent(config));
    const query = parsedConfig.query;
    const depth = parsedConfig.depth || 1;
    const max = parsedConfig.max || 1;
    const project = parsedConfig.project;
    await createTablesForProject(project);
    const urlsFile = path.join(ROOT_DOWNLOADS, project, 'scraped_urls.json');
    const processedFile = path.join(ROOT_DOWNLOADS, project, 'processed_urls.json');
    let socialLinks = [];
    let nonSocialLinks = [];
    if (fs.existsSync(urlsFile)) {
      const data = JSON.parse(fs.readFileSync(urlsFile, 'utf8'));
      socialLinks = data.socialLinks || [];
      nonSocialLinks = data.nonSocialLinks || [];
      console.log('Loaded scraped URLs from file');
    } else {
      const scraper = new Scraper(wss);
      await scraper.init();
      const queries = query.split(',').map(q => q.trim());
      let allLinks = [];
      let allImageLinks = [];
      for (const subquery of queries) {
        const links = await scraper.searchAndGetLinks(subquery);
        const imagelinks = await scraper.searchAndGetImageLinks(subquery);
        allLinks = allLinks.concat(links);
        allImageLinks = allImageLinks.concat(imagelinks);
      }
      const combinedLinks = [...new Set([...allLinks, ...allImageLinks])];
      console.warn('scraped links', combinedLinks);
      const classificationPromises = combinedLinks
        .filter(link => link.startsWith('http'))
        .map(async link => {
          const response = await prompt(
            'Classify if this URL is from a social media platform (Facebook, Twitter/X, Instagram, LinkedIn, etc.). Youtube and reddit is not considerd a social media platform. Return JSON: {"isSocialMedia": true/false}',
            link
          );
          return { link, isSocialMedia: response.isSocialMedia };
        });
      const results = await Promise.all(classificationPromises);
      results.forEach(({ link, isSocialMedia }) => {
        if (isSocialMedia) {
          socialLinks.push(link);
        } else {
          nonSocialLinks.push(link);
        }
      });
      if (!fs.existsSync(path.dirname(urlsFile))) {
        fs.mkdirSync(path.dirname(urlsFile), { recursive: true });
      }
      fs.writeFileSync(urlsFile, JSON.stringify({ socialLinks, nonSocialLinks }, null, 2));
      console.log('Saved scraped URLs to file');
      await scraper.close();
    }
    let processed = { social: [], nonSocial: [] };
    if (fs.existsSync(processedFile)) {
      processed = JSON.parse(fs.readFileSync(processedFile, 'utf8'));
    }
    let scraper = new Scraper(wss);
    await scraper.init();
    await scraper.processSocialLinks(socialLinks, query, depth, max, project, processed, processedFile);
    await scraper.close();
    scraper = new Scraper(wss);
    await scraper.init();
    await scraper.processNonSocialLinks(nonSocialLinks, query, project, processed, processedFile);
    await scraper.close();
    await loadPersonsData(project);
    res.status(200).json({ status: 'OK', message: 'Scraping complete', project });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});
app.post('/geoclip', (req, res) => {
  const { lat, lon, probability, image, detections, node_id, entity_idx, entity, timestamp } = req.body;
  if (!lat || !lon) {
    return res.status(400).send({ error: 'Missing required lat/lon fields.' });
  }
  const payload = { lat, lon, probability, image, detections, node_id, entity_idx, entity, timestamp };
  console.log(`Received GeoCLIP:`, JSON.stringify(payload, null, 2));
  wss.clients.forEach((client) => {
    if (client.readyState === 1) {
      client.send(JSON.stringify(payload));
    }
  });
  res.status(200).send('Broadcasted');
});
app.post('/graphdata', (req, res) => {
  const graphData = req.body;
  wss.clients.forEach((client) => {
    if (client.readyState === 1) {
      client.send(JSON.stringify(graphData));
    }
  });
  res.status(200).send('Graph data broadcasted');
});
app.post('/python-images', (req, res) => {
  const images = req.body.images; // Expecting an array of base64 strings without the data URI prefix
  console.log(images);
  wss.clients.forEach((client) => {
    if (client.readyState === 1) {
      client.send(JSON.stringify({ type: 'python_graph_images', images }));
    }
  });
  res.status(200).send('Python images broadcasted');
});
function stripHtmlTags(text) {
  return text.replace(/<[^>]*>?/gm, '');
}
async function filterAiText(text, model = "deepseek-chat") {
  const systemContent = `You are an AI detector that filters out likely AI-generated text from the given content. Analyze the text and return only the parts that appear to be human-generated. If the entire text seems AI-generated, return an empty string. Provide the filtered text in JSON: {"filtered_text": "..." }`;
  // Chunk the text if needed
  const chunks = [text]; // For simplicity, or split if too long
  const results = await processHtmlChunks(systemContent, chunks); // Reuse, even though not HTML
  // Assume results are array of {filtered_text: ...}
  const filtered = results.map(r => r.filtered_text || '').join('\n');
  return filtered;
}
app.post('/run-python', upload.single('face_image'), async (req, res) => {
  const analysesJSON = req.body.analyses;
  const project = req.body.project;
  if (!project) {
    return res.status(400).json({ error: 'Project not provided' });
  }
  if (!analysesJSON) {
    return res.status(400).json({ error: 'Analyses not provided' });
  }
  const analyses = JSON.parse(analysesJSON);
  const projectName = project.replace(/\s+/g, '_');
  let faceImageBase64 = null;
  let tempFilePath = null;

  // If an image file is uploaded, convert it to base64
  if (req.file) {
    faceImageBase64 = fs.readFileSync(req.file.path, 'base64');
    fs.unlinkSync(req.file.path); // Clean up the uploaded file
  }
  // Write base64 data to a temporary file if it exists
  if (faceImageBase64) {
    tempFilePath = path.join(ROOT_DOWNLOADS, `temp_face_image_${Date.now()}.txt`);
    try {
      fs.writeFileSync(tempFilePath, faceImageBase64);
      console.log(`Wrote base64 data to temporary file: ${tempFilePath}`);
    } catch (e) {
      console.error(`Error writing base64 to file: ${e}`);
      return res.status(500).json({ error: 'Failed to write base64 data to file' });
    }
  }

  // HTML to inner text conversion and AI filtering
  console.log("\nConverting HTML page_text to inner text and filtering AI-generated content...");
  const client = await pool.connect();
  try {
    const selectQuery = `SELECT id, content, page_text FROM "${projectName}_nodes" WHERE page_text IS NOT NULL AND page_text != '';`;
    const result = await client.query(selectQuery);
    const rows = result.rows;

    const limit = pLimit(15); // Limit concurrency to 5 to avoid rate limits and connection issues

    await Promise.all(rows.map(row => limit(async () => {
      const rowClient = await pool.connect();
      try {
        await rowClient.query('BEGIN');

        const node_id = row.id;
        let content_dict = typeof row.content === 'string' ? JSON.parse(row.content) : row.content;

        // Skip if already AI filtered
        if (content_dict.ai_filtered) {
          console.log(`Skipping node ${node_id} as already AI filtered`);
          return;
        }

        // Step 1: Strip HTML tags to get inner text
        let stripped_text = stripHtmlTags(row.page_text);

        // Step 2: Use processHtmlChunks to filter out AI-generated text
        const systemContent = `You are a text filter that removes website UI-related content (e.g., button labels, navigation menus, headers, footers, advertisements) from the given text. Retain only user-generated or content-rich text such as comments, blog posts, articles, and personal information (e.g., bios, profiles, descriptions). If no user-generated content is found, return an empty string. Provide the filtered text in JSON: {"filtered_text": "..."}`;
        const filtered_results = await processHtmlChunks(systemContent, [stripped_text]);
        const filtered_text = filtered_results.map(r => r.filtered_text || '').join('\n');

        // Update page_text if changed
        if (filtered_text !== row.page_text) {
          const updateQuery = `UPDATE "${projectName}_nodes" SET page_text = $1 WHERE id = $2;`;
          await rowClient.query(updateQuery, [filtered_text, node_id]);
          console.log(`Updated page_text for node ${node_id} with stripped HTML and AI filtered text`);
        }

        // Mark as AI filtered
        content_dict.ai_filtered = true;
        const update_content = JSON.stringify(content_dict);
        await rowClient.query(`UPDATE "${projectName}_nodes" SET content = $1 WHERE id = $2;`, [update_content, node_id]);
        console.log(`Marked node ${node_id} as AI filtered`);

        await rowClient.query('COMMIT');
      } catch (e) {
        await rowClient.query('ROLLBACK');
        console.error(`Error processing node ${row.id}: ${e}`);
      } finally {
        rowClient.release();
      }
    })));
  } catch (e) {
    console.error(`Error querying rows: ${e}`);
  } finally {
    client.release();
  }

  // Run Python script
  const analysesStr = analyses.join(',');
  const venvPython = '/home/a/Schreibtisch/OSINT-map/myenv/bin/python';
  const scriptPath = 'analysis.py';
  const args = [scriptPath, projectName, '--analyses', analysesStr];
  if (tempFilePath) {
    args.push('--face_image_file', tempFilePath); // Pass file path instead of base64
  }

  const pyProcess = spawn(venvPython, args);
  pyProcess.stdout.on('data', (data) => {
    io.emit('log', data.toString());
  });
  pyProcess.stderr.on('data', (data) => {
    console.error(`Python Error: ${data.toString()}`);
  });

  pyProcess.on('close', async (code) => {
    // Clean up the temporary file if it was created
    if (tempFilePath && fs.existsSync(tempFilePath)) {
      try {
        fs.unlinkSync(tempFilePath);
        console.log(`Deleted temporary file: ${tempFilePath}`);
      } catch (e) {
        console.error(`Error deleting temporary file: ${e}`);
      }
    }
    await loadPersonsData(projectName);
    res.send(`Analysis done with exit code ${code}`);
  });
});
server.listen(PORT, () => console.log(`Express running on :${PORT}`));
