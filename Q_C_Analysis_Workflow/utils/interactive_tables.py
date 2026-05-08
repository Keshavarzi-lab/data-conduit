'''
Helper functions for easily making interactive tables in notebooks. These are used in the design space exploration notebook, but could be used elsewhere too.

Contents
--------
- display_table
    Display a DataFrame as interactive table. Toggles include inline notebook vs separate browser tab, and saving the table as an HTML file.


'''

#===== Imports

import pandas as pd
import webbrowser
import html

from pathlib import Path
from tempfile import NamedTemporaryFile



def display_table(
        df: pd.DataFrame,
        *,
        inline: bool = True,
        in_browser: bool = False,
        save_html: bool = False,
        save_csv: bool = False,
        save_path: str | Path | None = None,
        title: str | None = None,
        include_datetime: bool = False,
        connected: bool = False,
        **datatable_kwargs,
    ) -> Path | None:
    '''
    Display a DataFrame as an interactive table. By default, the table is displayed inline in the notebook, 
    but can also be opened in a separate browser tab. Optionally, the table can be saved as an HTML file.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to display as an interactive table.
    inline : bool, default True
        Whether to display the table inline in the notebook. 
        If False, the table will not be displayed in the notebook, but can still be saved as an HTML file or opened in a browser.
    in_browser : bool, default False
        Whether to open the table in a separate browser tab. 
        If True, the table will be written to an HTML file and opened in the
        default web browser. If `save_html` is False, that browser file is
        temporary and `save_path` is still available for CSV output.
    save_html : bool, default False
        Whether to save the table as an HTML file. 
        If True, the table will be saved at `save_path` if provided, otherwise in a temporary file.
    save_csv : bool, default False
        Whether to save the DataFrame as a CSV file.
        If `save_path` is a directory, the CSV filename is derived from `title`.
        If `save_path` is an HTML filename, the CSV is saved next to it with a `.csv` suffix.
    save_path : str or Path, optional
        The path to save outputs if `save_html`, `save_csv`, or `in_browser` is True.
        This can be a directory or a complete file path.
        If not provided, a temporary file will be used for each saved output.
        Browser-only HTML files are always temporary unless `save_html` is True.
    title : str, optional
        An optional title to include in the HTML file. 
        This is ignored if `save_html` is False. 
        If None, title will be "Interactive Table {ss:mm::hh:dd}".
    include_datetime : bool, default False
        Whether to append the current date and time to a provided title.
        If `title` is None, the date and time are always included.
    connected : bool, default True
        Whether to use the connected version of DataTables (with separate JS/CSS files). 
        If False, the standalone version will be used (with all JS/CSS embedded in the HTML). 
        The connected version results in smaller HTML files and faster loading, but requires an internet connection to load the JS/CSS from a CDN.
    **datatable_kwargs
        Additional keyword arguments to pass to `itables.to_html_datatable()`. 
        See the itables documentation for available options.
    
    Returns
    -------
    Path or None
        The path to the saved HTML file if one is created, otherwise the CSV path
        if only `save_csv` is True. Returns None if no file is saved.
    '''
    from itables import to_html_datatable

    #=== Generate HTML for the DataFrame using itables
    table_html = to_html_datatable(
                                df,
                                connected=connected,
                                **datatable_kwargs,
                            )
    
    #=== Build the full HTML document
    timestamp = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}"
    if title is None:
        title = f"Interactive Table {timestamp}"
    elif include_datetime:
        title = f"{title} {timestamp}"

    safe_title = "".join(
        char.lower() if char.isalnum() else "_"
        for char in title
    ).strip("_") or "interactive_table"

    # Escape only the HTML-facing copy. The raw title is still useful for
    # deriving readable filenames when save_path points to a directory.
    html_title = html.escape(title)

    full_html = f"""<!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{html_title}</title>
    </head>
    <body>
        {table_html}
    </body>
    </html>
    """
    
    #=== Display the table inline if requested
    if inline:
        from IPython.display import HTML, display

        display(HTML(table_html))

    def _output_path(path: str | Path | None, suffix: str) -> Path:
        if path is None:
            with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                return Path(tmp_file.name)

        output_path = Path(path)

        # Accept either a complete file path or a directory. A path without a
        # suffix is treated as a directory, even if it does not exist yet.
        if output_path.is_dir() or output_path.suffix == "":
            output_path = output_path / f"{safe_title}{suffix}"
        elif output_path.suffix != suffix:
            output_path = output_path.with_suffix(suffix)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    #=== Save the HTML to a file if requested
    html_path = None
    if save_html:
        html_path = _output_path(save_path, ".html")
        html_path.write_text(full_html, encoding="utf-8")

    #=== Open the table in a browser if requested
    if in_browser:
        browser_html_path = html_path
        if browser_html_path is None:
            # Browser display needs an HTML file, but this should not consume
            # save_path unless the caller explicitly asked to save HTML.
            browser_html_path = _output_path(None, ".html")
            browser_html_path.write_text(full_html, encoding="utf-8")

        webbrowser.open(browser_html_path.resolve().as_uri())

    #=== Save the raw DataFrame to CSV if requested
    csv_path = None
    if save_csv:
        csv_path = _output_path(save_path, ".csv")
        df.to_csv(csv_path, index=False)

    return html_path or csv_path
