import asyncio
from typing import Optional
from frontend.meme_bot import MemeBot

bot: Optional[MemeBot] = None


async def main():
    """doc"""
    global bot
    bot = MemeBot()
    await bot.start()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
