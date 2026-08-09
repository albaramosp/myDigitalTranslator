from application.use_cases.extract_web_text import ExtractWebText
from application.use_cases.translate_web_text import TranslateWebText
from domain.prompts.autism_adapter_prompt import AutismAdapterPrompt
from infrastructure.llm.llm_factory import LlmFactory
from infrastructure.requests_web_downloader import RequestsWebDownloader
from infrastructure.trafilatura_parser import TrafilaturaWebAdapter
from logging_config import setup_logging
if __name__ == '__main__':
    setup_logging()

    urls = [
        # 1. Simplifying chaotic digital environments
        "https://www.amazon.com/Milk-Makeup-Primer-hialur%C3%B3nico-niacinamida/dp/B07PQN4XZ1/ref=sr_1_3_sspa?__mk_es_US=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=2IHY4GU262J1Z&dib=eyJ2IjoiMSJ9.saoxDPi7hNIoXgmYvdzoplCwFckV1qzeIvCY0Px60twtTzBRu_emUW7DEe3LPckN6NPSRJkuM3cT2kp3i2tdJbNxLPnwOOcChxLAmifwqZMfDBBemdmleBIxsdtps5ba0FpnYl_fvmEF5Lto3-GcOW5XwDEMRV3FxNV8pQIXvC3BCnsk-eS9vkCYuHQ4H6WemY1f9sv18ctdNseE--FZ3K5FPGILSfIv7GAL_RbsIKcWlz7-zOTHmzZklPipMI4FCi9Ofw1txI3DsYAopBU3C1gYPvoRa8Keq3zjZBaGF7Q.O6uQxN67gIM2lGMAFqv1sZt9xt9-rHOwhN23Vu6M_w0&dib_tag=se&keywords=maquillaje&qid=1785155599&sprefix=maquilla%2Caps%2C266&sr=8-3-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1",
        "https://polymarketanalytics.com/event/jesus-christ-return-before-2027-odds-5-monday-night-12-1-am",

        # 2. Removing ambiguous or implicit language
        "https://foro.portalpez.com/threads/novata-con-muchas-dudas.111466/#post-1084076",

        # 3. Planning and anticipating events
        "https://www.conociendoitalia.com/como-organizar-un-viaje-a-roma/",

        # 4. Filtering news and alarmist information
        "https://www.20minutos.es/nacional/incendios-madrid-avila-toledo-castellon-espana-directo-ultima-hora-fuego-hectareas-confinamientos-evacuaciones_7017956_6.html",
    ]

    uc = ExtractWebText(
        downloader=RequestsWebDownloader(),
        adapter=TrafilaturaWebAdapter()
    )

    for url in urls:
        text = uc.execute(url)
        if text:
            adapted_text = TranslateWebText(
                LlmFactory.create(),
                AutismAdapterPrompt()
            ).execute(text)

            print(adapted_text)
            print("***************")




