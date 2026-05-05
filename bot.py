"""
CSBuy Cat Game Bot — v2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Descobertas do vídeo de gameplay:
  • O jogo é um POPUP na homepage (não URL separada)
  • 4 ações por sessão: alimentar, corações, Collect (máquina), Daily Draw
  • Corações são vermelhos sólidos dentro do popup
  • Daily Draw dá +200g por dia
  • Botão laranja "Collect" aparece quando máquina está pronta
  • Viewport desktop (não mobile) — jogo roda no site desktop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import asyncio
import io
import json
import logging
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

# ── Config ────────────────────────────────────────────────────────────────────
CSSBUY_URL     = "https://www.cssbuy.com"
COOKIES_JSON   = os.environ["CSSBUY_COOKIES"]

VIEWPORT_W     = 1280
VIEWPORT_H     = 800

HEART_DURATION = 4 * 60   # segundos caçando corações por execução
HEART_INTERVAL = 1.5      # segundos entre cada scan

FEED_TIMES     = 5        # quantas vezes alimenta por sessão

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


# ── BBox helper ───────────────────────────────────────────────────────────────

class BBox:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def contains(self, px, py, margin=10):
        return (self.x - margin <= px <= self.x + self.w + margin and
                self.y - margin <= py <= self.y + self.h + margin)

    def __repr__(self):
        return f"BBox(x={self.x:.0f}, y={self.y:.0f}, w={self.w:.0f}, h={self.h:.0f})"


# ── Visão computacional ────────────────────────────────────────────────────────

def _crop_to_bbox(img: np.ndarray, bbox: BBox | None):
    if bbox is None:
        return img, 0, 0
    x1 = max(0, int(bbox.x))
    y1 = max(0, int(bbox.y))
    x2 = min(img.shape[1], int(bbox.x + bbox.w))
    y2 = min(img.shape[0], int(bbox.y + bbox.h))
    return img[y1:y2, x1:x2], x1, y1


def detect_hearts(screenshot_bytes: bytes, bbox: BBox | None = None) -> list[tuple[int, int]]:
    """
    Detecta corações vermelhos/rosa no screenshot.
    Restringe a busca ao bbox do popup se fornecido.
    Retorna lista de (x, y) em coordenadas absolutas da página.
    """
    img = np.array(Image.open(io.BytesIO(screenshot_bytes)))
    roi, ox, oy = _crop_to_bbox(img, bbox)
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    ranges = [
        (np.array([0,   150, 100]), np.array([10,  255, 255])),   # vermelho 1
        (np.array([160, 150, 100]), np.array([180, 255, 255])),   # vermelho 2
        (np.array([140,  80, 120]), np.array([165, 255, 255])),   # rosa
    ]
    mask = sum(cv2.inRange(hsv, lo, hi) for lo, hi in ranges)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clicks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 150 < area < 10_000:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            clicks.append((int(M["m10"] / M["m00"]) + ox,
                            int(M["m01"] / M["m00"]) + oy))

    if clicks:
        log.info(f"❤️  Corações detectados: {len(clicks)} → {clicks}")
    return clicks


def detect_orange_collect(screenshot_bytes: bytes, bbox: BBox | None = None) -> tuple[int, int] | None:
    """
    Detecta o botão laranja 'Collect' visualmente.
    Retorna (x, y) do centro ou None.
    """
    img = np.array(Image.open(io.BytesIO(screenshot_bytes)))
    roi, ox, oy = _crop_to_bbox(img, bbox)
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(hsv, np.array([10, 180, 150]), np.array([25, 255, 255]))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500 and area > best_area:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            best = (int(M["m10"] / M["m00"]) + ox,
                    int(M["m01"] / M["m00"]) + oy)
            best_area = area

    if best:
        log.info(f"🟠 Botão Collect detectado em {best}")
    return best


def save_debug(screenshot_bytes: bytes, tag: str):
    path = f"screenshot_{tag}_{int(time.time())}.png"
    with open(path, "wb") as f:
        f.write(screenshot_bytes)
    log.info(f"📸 {path}")


# ── Login ──────────────────────────────────────────────────────────────────────

async def do_login(context, page):
    log.info("Injetando cookies…")
    cookies = json.loads(COOKIES_JSON)
    for c in cookies:
        if not c.get("domain"):
            c["domain"] = ".cssbuy.com"
    await context.add_cookies(cookies)
    log.info(f"✅ {len(cookies)} cookies injetados.")

    await page.goto(CSSBUY_URL, wait_until="networkidle")
    await asyncio.sleep(2)

    if "login" in page.url.lower():
        raise RuntimeError("❌ Cookies expirados! Gere novos com inspect_game.py")
    log.info("Sessão válida ✓")


# ── Localizar popup ────────────────────────────────────────────────────────────

async def wait_for_popup(page, timeout=20) -> BBox | None:
    """Espera o popup do jogo aparecer e retorna seu BBox."""
    log.info("Aguardando popup do jogo…")
    selectors = [
        ".pet-game-modal", ".pet-game", ".cat-game", "#pet-game",
        "[class*='pet'][class*='game']", "[class*='cat'][class*='game']",
        ".game-modal", ".game-popup", "[class*='game-wrap']",
        "[class*='turbo']", ".modal-content", ".modal.show .modal-body",
    ]
    deadline = time.time() + timeout
    while time.time() < deadline:
        for sel in selectors:
            try:
                el = await page.query_selector(sel)
                if el and await el.is_visible():
                    bb = await el.bounding_box()
                    if bb and bb["width"] > 80 and bb["height"] > 150:
                        bbox = BBox(bb["x"], bb["y"], bb["width"], bb["height"])
                        log.info(f"✅ Popup via '{sel}': {bbox}")
                        return bbox
            except Exception:
                pass
        await asyncio.sleep(1)
    log.warning("⚠️ Popup não detectado pelo DOM — sem restrição de área.")
    return None


async def click_cat_to_open_game(page) -> bool:
    """
    Clica no ícone do gato no canto superior esquerdo da homepage
    para abrir o popup do jogo ("Grow me, win prizes!").
    """
    log.info("Procurando ícone do gato…")

    cat_selectors = [
        ":has-text('Grow me')",
        ":has-text('win prizes')",
        "img[src*='cat']",
        "img[src*='pet']",
        "img[src*='mascot']",
        "[class*='pet-icon']",
        "[class*='cat-icon']",
        "[class*='mascot']",
        "[class*='pet-entry']",
        "[class*='game-entry']",
        "[class*='grow']",
    ]

    for sel in cat_selectors:
        try:
            el = await page.query_selector(sel)
            if el and await el.is_visible():
                await el.click()
                log.info(f"✅ Gato clicado via '{sel}'")
                await asyncio.sleep(2)
                return True
        except Exception:
            pass

    # Fallback por posição — gato fica em ~(85, 180) num viewport 1280x800
    log.warning("Seletores não encontraram o gato — tentando posição aproximada…")
    try:
        await page.mouse.click(85, 180)
        await asyncio.sleep(2)
        save_debug(await page.screenshot(), "cat_click_fallback")
        return True
    except Exception as e:
        log.error(f"Falha ao clicar no gato: {e}")
        return False


async def get_game_context(page):
    """
    1. Clica no gato para abrir o popup.
    2. Retorna (ctx, bbox) — ctx é o frame do jogo ou a page.
    """
    await click_cat_to_open_game(page)

    for frame in page.frames:
        if any(k in frame.url for k in ["pet", "game", "turbo", "cat"]):
            log.info(f"Iframe do jogo: {frame.url}")
            for iframe_el in await page.query_selector_all("iframe"):
                src = await iframe_el.get_attribute("src") or ""
                if any(k in src for k in ["pet", "game", "turbo"]):
                    bb = await iframe_el.bounding_box()
                    if bb:
                        return frame, BBox(bb["x"], bb["y"], bb["width"], bb["height"])
            return frame, None

    bbox = await wait_for_popup(page)
    return page, bbox


# ── Ações do jogo ─────────────────────────────────────────────────────────────

async def safe_click(ctx, selector: str, timeout=4000) -> bool:
    try:
        await ctx.click(selector, timeout=timeout)
        log.info(f"  ✓ {selector}")
        await asyncio.sleep(0.5)
        return True
    except (PlaywrightTimeout, Exception):
        return False


async def close_overlay(ctx):
    """Fecha qualquer overlay/popup secundário (Congrats, anúncio, etc.)."""
    for sel in ["button.close", ".close-btn", "[class*='close']",
                ".modal-close", ".congrats-close", ".reward-close",
                "button[aria-label='Close']"]:
        try:
            el = await ctx.query_selector(sel)
            if el and await el.is_visible():
                await el.click()
                log.info(f"Overlay fechado: {sel}")
                await asyncio.sleep(0.8)
                return True
        except Exception:
            pass
    try:
        await ctx.keyboard.press("Escape")
    except Exception:
        pass
    return False


async def do_daily_draw(ctx) -> bool:
    """Executa o Daily Fortune Draw (+200g observado no vídeo)."""
    log.info("🎰 Daily Draw…")
    opened = False
    for sel in ["[class*='daily-draw']", "[class*='daily'][class*='draw']",
                "img[src*='daily']", ".daily-draw", ":has-text('Daily Draw')"]:
        if await safe_click(ctx, sel, timeout=3000):
            opened = True
            await asyncio.sleep(1.5)
            break

    if not opened:
        log.warning("Daily Draw não encontrado.")
        return False

    for sel in ["button:has-text('Draw now')", "button:has-text('Draw')",
                "[class*='draw-btn']", ".draw-now"]:
        if await safe_click(ctx, sel, timeout=4000):
            log.info("✅ Daily Draw feito!")
            await asyncio.sleep(2)
            await close_overlay(ctx)
            return True

    await close_overlay(ctx)
    return False


async def do_blind_box(ctx) -> bool:
    """Tenta abrir o Blind Box diário (canto direito do popup)."""
    log.info("📦 Blind Box…")
    for sel in ["[class*='blind-box']", "[class*='blindbox']",
                "img[src*='blind']", ":has-text('Blind Box')", ".blind-box"]:
        if await safe_click(ctx, sel, timeout=3000):
            await asyncio.sleep(1.5)
            await close_overlay(ctx)
            log.info("✅ Blind Box aberto!")
            return True
    log.info("Blind Box indisponível.")
    return False


async def collect_machine(ctx) -> bool:
    """Coleta ração da máquina quando o timer acabou."""
    log.info("⚙️  Máquina de comida…")
    for sel in ["[class*='collect']", "button:has-text('Collect')", ".collect-btn",
                "[class*='machine']", "[class*='production']", "img[src*='machine']"]:
        if await safe_click(ctx, sel, timeout=3000):
            log.info("✅ Máquina coletada!")
            await asyncio.sleep(1)
            return True
    log.info("Máquina ainda carregando.")
    return False


async def feed_cat(ctx, times: int = FEED_TIMES) -> int:
    """Clica no saco de ração N vezes."""
    log.info(f"🍖 Alimentando ({times}x)…")
    selectors = [
        "[class*='food'][class*='bag']", "[class*='feedbag']", "[class*='feed-bag']",
        "img[src*='food_bag']", "img[src*='feedbag']", "img[src*='bag']",
        ".feed-btn", "[class*='feed']", "button:has-text('Feed')",
    ]
    fed = 0
    for _ in range(times):
        for sel in selectors:
            if await safe_click(ctx, sel, timeout=3000):
                fed += 1
                await asyncio.sleep(0.8)
                break
        await asyncio.sleep(0.4)

    log.info(f"Alimentou {fed}/{times}x.")
    return fed


async def hunt_hearts(page, ctx, bbox: BBox | None) -> int:
    """
    Loop principal: caça corações por HEART_DURATION segundos.
    A cada 30s também tenta detectar o botão Collect laranja visualmente.
    """
    log.info(f"🏹 Caçando corações por {HEART_DURATION // 60}min…")
    end_time   = time.time() + HEART_DURATION
    collect_cd = 0
    total      = 0

    while time.time() < end_time:
        ss = await page.screenshot()

        # Corações
        for (x, y) in detect_hearts(ss, bbox):
            try:
                await page.mouse.click(x, y)
                total += 1
                await asyncio.sleep(0.25)
            except Exception as e:
                log.warning(f"Erro clique coração ({x},{y}): {e}")

        # Botão Collect laranja (visual, a cada 30s)
        if time.time() > collect_cd:
            pos = detect_orange_collect(ss, bbox)
            if pos:
                try:
                    await page.mouse.click(*pos)
                    log.info("🟠 Collect clicado!")
                    await asyncio.sleep(1)
                except Exception:
                    pass
            collect_cd = time.time() + 30

        await asyncio.sleep(HEART_INTERVAL)

    log.info(f"Total corações: {total}")
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    log.info("=" * 55)
    log.info("  CSBuy Cat Bot v2")
    log.info("=" * 55)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage",
                  "--disable-blink-features=AutomationControlled"]
        )
        context = await browser.new_context(
            viewport={"width": VIEWPORT_W, "height": VIEWPORT_H},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="zh-CN",
        )
        page = await context.new_page()

        try:
            # 1. Login
            await do_login(context, page)
            save_debug(await page.screenshot(), "01_login")

            # 2. Localizar popup / iframe
            ctx, bbox = await get_game_context(page)
            log.info(f"ctx={'iframe' if ctx != page else 'page'}, bbox={bbox}")
            save_debug(await page.screenshot(), "02_popup")

            # 3. Fechar overlays iniciais
            await close_overlay(ctx)
            await asyncio.sleep(1)

            # 4. Daily Draw
            await do_daily_draw(ctx)
            await asyncio.sleep(1)

            # 5. Blind Box
            await do_blind_box(ctx)
            await asyncio.sleep(1)

            # 6. Coletar máquina
            await collect_machine(ctx)
            await asyncio.sleep(1)

            # 7. Alimentar
            await feed_cat(ctx, FEED_TIMES)
            await asyncio.sleep(1)

            # 8. Caçar corações (loop principal)
            await hunt_hearts(page, ctx, bbox)

            save_debug(await page.screenshot(), "03_final")

        except Exception as e:
            log.error(f"💥 {e}", exc_info=True)
            try:
                save_debug(await page.screenshot(), "error")
            except Exception:
                pass
            raise

        finally:
            await browser.close()
            log.info("Encerrado ✓")


if __name__ == "__main__":
    asyncio.run(main())
