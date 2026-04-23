from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


_run_config = CrawlerRunConfig(
    markdown_generator=DefaultMarkdownGenerator(),
    cache_mode=CacheMode.BYPASS,
)


async def html_to_markdown(html_content: str, crawler: AsyncWebCrawler) -> str:
    result = await crawler.arun(url=f"raw:{html_content}", config=_run_config)
    if not result.success:
        raise Exception(f"Failed to convert HTML to Markdown: {result.error_message}")
    md_obj = result.markdown
    if hasattr(md_obj, 'raw_markdown'):
        return md_obj.raw_markdown
    return str(md_obj)