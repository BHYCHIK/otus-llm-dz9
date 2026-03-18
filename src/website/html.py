def get_main_page_html():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Image search</title>
    </head>
    <body>
        <h1>Welcome to image searcher!</h1>
    <form method="get" action="find_images" enctype=multipart/form-data>
        Поисковый запрос: <input type="text" name="image_description" /><br>
        Сколько картинок ищем? <input type="number" name="top_n" value="5" /><br>
        <input type="submit" value="Search" />
    </form>
    </body>
    </html>
"""

def get_serp_page_html(image_links):
    images = '<br>'.join([f"<img src=\"{link}\" width=\"200\" height=\"150\">" for link in image_links])
    print(images)

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Image search result</title>
    </head>
    <body>
    {images}
    <form method="get" action="/">
        <br><input type="submit" name="back" value="back" /><br>
    </form>
    </body>
    </html>
    """