import asyncio
from typing import Optional
from frontend.meme_bot import MemeBot

bot: Optional[MemeBot] = None


async def main(debug=True):
    """doc"""
    global bot
    bot = MemeBot(debug=debug)
    await bot.start()


def start():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    start()
