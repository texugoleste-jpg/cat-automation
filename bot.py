"""
CSBuy Cat Game Bot
- Faz login no CSBuy
- Alimenta o gato
- Detecta e clica nos corações (via visão computacional)
- Coleta comida da máquina quando disponível
"""

import asyncio
import os
import sys
import time
import logging
import numpy as np
import cv2
from PIL import Image
import io
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

# ── Config ──────────────────────────────────────────────────────────────────
CSSBUY_URL   = "https://www.cssbuy.com"
GAME_URL     = "https://www.cssbuy.com/pet"   # ⚠️ ajuste para a URL real do jogo
COOKIES_JSON = os.environ["CSSBUY_COOKIES"]   # JSON exportado do browser

HEART_SCAN_DURATION = 5 * 60   # segundos caçando corações (5 min)
HEART_SCAN_INTERVAL = 2        # segundos entre cada scan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


# ── Detecção de corações via OpenCV ─────────────────────────────────────────

def detect_hearts(screenshot_bytes: bytes) -> list[tuple[int, int]]:
    """
    Recebe screenshot em bytes, devolve lista de (x, y) com centros de corações detectados.
    Corações no jogo são rosa/vermelho brilhante — filtramos por HSV.
    """
    img = np.array(Image.open(io.BytesIO(screenshot_bytes)))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Faixa de cor rosa/vermelho dos corações
    # Ajuste lower/upper se necessário inspecionando a cor real
    masks = []
    for (lower, upper) in [
        (np.array([0,   120, 120]), np.array([10,  255, 255])),   # vermelho 1
        (np.array([160, 120, 120]), np.array([180, 255, 255])),   # vermelho 2
        (np.array([140,  80, 120]), np.array([170, 255, 255])),   # rosa
    ]:
        masks.append(cv2.inRange(hsv, lower, upper))

    mask = masks[0] | masks[1] | masks[2]

    # Remove ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Encontra contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clicks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 200 < area < 8000:      # filtra tamanho (~coração)
            M  = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            clicks.append((cx, cy))

    log.info(f"Corações detectados: {len(clicks)} → {clicks}")
    return clicks


def save_debug_screenshot(screenshot_bytes: bytes, tag: str):
    path = f"screenshot_{tag}_{int(time.time())}.png"
    with open(path, "wb") as f:
        f.write(screenshot_bytes)
    log.info(f"Screenshot salvo: {path}")


# ── Helpers de página ────────────────────────────────────────────────────────

async def safe_click(page, selector: str, timeout=5000) -> bool:
    try:
        await page.click(selector, timeout=timeout)
        log.info(f"Clicou em: {selector}")
        return True
    except PlaywrightTimeout:
        log.warning(f"Elemento não encontrado: {selector}")
        return False


async def get_game_frame(page):
    """
    O jogo pode estar dentro de um <iframe>.
    Devolve o frame correto ou a própria page se não houver iframe.
    """
    frames = page.frames
    for frame in frames:
        url = frame.url
        if "pet" in url or "game" in url or "turbolink" in url.lower():
            log.info(f"Frame do jogo encontrado: {url}")
            return frame
    return page   # fallback: sem iframe


# ── Ações do jogo ────────────────────────────────────────────────────────────

async def do_login(context, page):
    """
    Injeta cookies salvos no GitHub Secret em vez de fazer login manual.
    Evita CAPTCHA, 2FA e mudanças no formulário de login.
    """
    import json
    log.info("Injetando cookies…")

    cookies = json.loads(COOKIES_JSON)

    # Garante que o domínio está presente em todos os cookies
    for c in cookies:
        if "domain" not in c or not c["domain"]:
            c["domain"] = ".cssbuy.com"

    await context.add_cookies(cookies)
    log.info(f"{len(cookies)} cookies injetados.")

    # Visita a home para validar sessão
    await page.goto(CSSBUY_URL, wait_until="networkidle")

    # Verifica se ainda está logado
    if "login" in page.url.lower():
        raise RuntimeError("❌ Cookies expirados! Gere novos cookies com inspect_game.py")

    log.info("Sessão válida ✓")


async def navigate_to_game(page):
    log.info(f"Navegando para o jogo: {GAME_URL}")
    await page.goto(GAME_URL, wait_until="networkidle")
    await asyncio.sleep(3)   # aguarda animações iniciais


async def feed_cat(frame):
    """
    Clica no ícone de alimentar (saco de ração 740g).
    ⚠️ Ajuste o seletor conforme inspeção do DOM real.
    """
    log.info("Tentando alimentar o gato…")
    selectors = [
        ".feed-btn",
        "[class*='feed']",
        "[class*='food']",
        "img[src*='food']",
        "img[src*='bag']",
    ]
    for sel in selectors:
        if await safe_click(frame, sel, timeout=3000):
            await asyncio.sleep(1)
            return True
    log.warning("Botão de alimentar não encontrado — verifique o seletor.")
    return False


async def collect_food_machine(frame):
    """
    Coleta a comida produzida pela máquina (contador recarrega a cada 1h).
    ⚠️ Ajuste o seletor conforme inspeção do DOM real.
    """
    log.info("Verificando máquina de comida…")
    selectors = [
        ".machine",
        "[class*='machine']",
        "[class*='production']",
        "img[src*='machine']",
    ]
    for sel in selectors:
        if await safe_click(frame, sel, timeout=3000):
            await asyncio.sleep(1)
            log.info("Comida coletada da máquina!")
            return True
    log.warning("Máquina não encontrada — verifique o seletor.")
    return False


async def hunt_hearts(page, frame, duration: int = HEART_SCAN_DURATION):
    """
    Durante `duration` segundos, tira screenshots e clica nos corações detectados.
    """
    log.info(f"Iniciando caça aos corações por {duration}s…")
    end_time = time.time() + duration
    total_clicks = 0

    while time.time() < end_time:
        screenshot = await page.screenshot()
        hearts = detect_hearts(screenshot)

        for (x, y) in hearts:
            try:
                await page.mouse.click(x, y)
                total_clicks += 1
                log.info(f"  ❤️  Coração clicado em ({x}, {y})")
                await asyncio.sleep(0.3)
            except Exception as e:
                log.warning(f"Erro ao clicar em ({x}, {y}): {e}")

        await asyncio.sleep(HEART_SCAN_INTERVAL)

    log.info(f"Caça encerrada. Total de corações clicados: {total_clicks}")
    return total_clicks


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    log.info("=" * 50)
    log.info("CSBuy Cat Bot iniciado")
    log.info("=" * 50)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        context = await browser.new_context(
            viewport={"width": 390, "height": 844},   # simula iPhone 14
            user_agent=(
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
            )
        )
        page = await context.new_page()

        try:
            # 1. Injetar cookies (login)
            await do_login(context, page)

            # 2. Ir para o jogo
            await navigate_to_game(page)

            # Salva screenshot inicial para debug
            ss = await page.screenshot()
            save_debug_screenshot(ss, "after_navigate")

            # 3. Obtém o frame correto (se houver iframe)
            frame = await get_game_frame(page)

            # 4. Alimentar o gato
            await feed_cat(frame)

            # 5. Coletar comida da máquina
            await collect_food_machine(frame)

            # 6. Caçar corações
            await hunt_hearts(page, frame)

        except Exception as e:
            log.error(f"Erro inesperado: {e}", exc_info=True)
            try:
                ss = await page.screenshot()
                save_debug_screenshot(ss, "error")
            except:
                pass
            raise

        finally:
            await browser.close()
            log.info("Browser fechado. Bot encerrado.")


if __name__ == "__main__":
    asyncio.run(main())
