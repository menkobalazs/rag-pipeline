
import wikipedia # type: ignore
import wikipediaapi # type: ignore


search_results = wikipedia.search("KPMG",)
print(search_results[:5], len(search_results))

wiki_wiki = wikipediaapi.Wikipedia(user_agent='user@nowhere.com', language='en')
page = wiki_wiki.page(search_results[0])
print(page.fullurl)
print(page.text[:1000])