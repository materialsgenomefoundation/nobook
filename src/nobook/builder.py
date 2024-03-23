"""treat documentation as data

this builder exists to provide more control to site authors when working with notebook content.
sphinx can use notebooks as source, but their reliable structure is destroyed in favor of
a docutils representation. no one wants that. mkdocs can integrate notebooks with their plugin system, 
but then you need to consider mkdocs conventions (eg. `markdown` attributes) when authoring content.
authors should be able to write naturally without referring to document, if something needs to change
then we edit it in post.

this builder is build off of nbconvert, the default package for exporting notebooks.
the notebook file format is a powerful way to store hypermedia with metadata.
we can use it represent ANY file type and we exploit that fact to acheive our documentation goals.

the notebook schema forms the backbone for this approach.
a key design factor for this approach is that ALL text based documents are treated as notebooks.
the notebook file format is an ideal representation of text files that require metadata.

"""

from enum import Enum
from functools import lru_cache
import re, bs4
import nbconvert
import nbformat
from pandas import DataFrame, concat
from numpy import vectorize
from pandas import Index, Series
from asyncio import gather
from dataclasses import dataclass, field
from anyio import Path
from toolz import compose_left as compose
import nobook.utils

MIDGY = re.compile("^\%\%[\s+,(pidgy),(midgy)]")


class ConfigKind(Enum):
    mkdocs = "mkdocs"
    jb = "_toc"
    sphinx = "conf"
    pyproject = "pyproject"
    none = "none"

    @classmethod
    def guess(cls, sequence):
        for member, value in cls.__members__.items():
            if value.value in sequence:
                return value
        return cls.none


@dataclass
class Config:
    root: None = field(default_factory=Path.cwd)
    toc: None = None
    include: None = None
    exclude: None = None
    target: None = "site/draft"
    configs: None = None
    kind: ConfigKind = None
    execute: bool = False

    def __post_init__(self):
        import pathspec

        if isinstance(self.exclude, str):
            self.exclude = self.exclude.splitlines()
        if isinstance(self.exclude, list | tuple | set):
            self.exclude = pathspec.PathSpec.from_lines(pathspec.GitIgnorePattern, self.exclude)

        for key in ("toc", "root"):
            value = getattr(self, key)
            if isinstance(value, str):
                setattr(self, key, Path(value))

    @classmethod
    def new(cls, *args, **kwargs):
        df = Contents()
        df.config = cls(*args, **kwargs)
        return df

    async def load_config(self):
        """load configuration files to style project"""
        # this projects exists to create alternative accessible representations
        # of existing documentation. we should be able to consume configurations
        # from other static site generators like jupyter book or mkdocs
        if self.kind == ConfigKind.jb and self.toc:
            df = await self._load_config_jb(self, self.toc)
        elif self.kind == ConfigKind.mkdocs:
            raise NotImplementedError("mkdocs discovery is no implemented yet")
        elif self.kind == ConfigKind.pyproject:
            raise NotImplementedError("pyproject discovery is no implemented yet")
        elif self.kind == ConfigKind.sphinx:
            raise NotImplementedError("sphinx discovery is no implemented yet")
        else:
            df = await self._load_config_none(self)
        if self.exclude:
            df = df[~df.index.map(self.exclude.match_file)]
        return df

    async def _load_config_none(self, config):
        """the default discovery techniques for the contents"""
        from pandas import concat

        return DataFrame(
            index=concat(
                (
                    Index([config.root]).path().glob(include, recursive=True)
                    for include in config.include
                ),
                axis=1,
            )["file"]
            .pipe(Index, name="file")
            .apath()
        )

    async def _load_config_jb(self, config, toc="_toc.yml"):
        """load an index of files from a jupyter book table of contents."""
        # this doesnt actually use the file. probably should
        toc = (await Index([config.root / toc], name="path").apath().apath.load()).series()
        chapters = toc.parts.enumerate("chapter").series()
        sections = chapters.chapters.enumerate("section").series()
        files = (
            sections.sections.dropna()
            .enumerate("subsection")
            .series()
            .combine_first(
                sections[["file"]].set_index(
                    Index([0] * len(sections), name="subsection"), append=True
                )
            )["file"]
        )
        files = files.apath().apply(config.root.__truediv__)
        return files.to_frame().reset_index().set_index("file")


class _ContentBase:
    """a base class for the contents series and dataframes."""

    api = config = cells = outputs = bundles = None
    _metadata = ["api", "config", "cells", "outputs", "bundles", "headings"]
    Config = Config


class ContentsSeries(_ContentBase, Series):
    """ContentsSeries exists to carry _metadata through slicing and expanding the dataframe."""

    @property
    def _constructor(self):
        return ContentsSeries

    @property
    def _constructor_expanddim(self):
        return Contents


class Contents(_ContentBase, DataFrame):
    """a Contents from carries documentation information about files using the nbformat as an interface.
    the"""

    @property
    def _constructor(self):
        return Contents

    @property
    def _constructor_sliced(self):
        return ContentsSeries

    async def expand(self):
        if not len(self):
            self = await self.discover()
        self = self.expand_index()
        self = await self.load_contents()
        self = self.expand_contents()
        self = await self.expand_features()
        return self

    async def compact(self, export=False):
        await gather(self.template_contents(), self.template_index())
        if export:
            await self.export()
        return self

    async def discover(self):
        """discover the files defined in the configuration"""

        index = await self.config.load_config()
        self = self.reindex(index.index)
        self.update(index)
        return self

    def expand_index(self):
        return self

    async def load_contents(self, key="contents"):
        """load the contents from the index and extract the subsequent frames for the cells, outputs, and mimebundles"""

        # this will only load the files in the index.
        # down selecting ahead of time can speed up development efforts
        if "sha" not in self:
            self.loc[:, "sha"] = self.loc[:, key] = None
        self.update(
            await self.index[await self.index.apath().apath.exists()]
            .apath()
            .apath.notebook(sha=True, key=key)
        )
        return self

    def expand_contents(self):
        """expand the contents in structured frames of the cells, outputs and mimebundles"""
        # prepare_notebook_contents is an in place operation on the document contents
        self.contents.dropna().apply(prepare_notebook_contents)

        # preprocess notebooks to make changes to shape and structure of the documents
        # optimizations could be performed on the outputs bundles to improve memory storage
        # for all the dataframes that are made.

        # self = self.apply(self._preprocess_contents, axis=1)

        # a notebook document is a hierarhical document that easiest to work through
        # multiple dataframes.
        # 1. a frame indexed by each of the cells
        self.cells = self["contents"].itemgetter("cells").enumerate("cell").series()
        # 2. a frame indexed by each of the outputs
        if "outputs" in self.cells:
            self.outputs = (
                self.cells.pop("outputs").dropna().enumerate("output").dropna(how="all").series()
            )
            if "data" in self.outputs:
                # 3. a frame indexed by each of the output mimebundles
                self.bundles = self.outputs.pop("data").dropna().series()
        return self

    async def expand_features(self):
        self.pipe(set_target_files)
        self.pipe(render_markdown_contents)
        self.pipe(replace_html_attachments)
        self.pipe(update_notebooks_with_render_contents)
        self.pipe(collect_contents_headings_title)
        self.pipe(collect_contents_headings)
        self.pipe(collect_site_navigation)
        return self

    async def template(self):
        await gather(self.template_contents(), self.template_index())

    async def template_contents(self):
        if "html" not in self:
            self["html"] = None
        # could expose resources in the way the markdown env is
        html = (
            await self.dropna(subset="contents")
            .apply(
                lambda s: export_from_notebook_node(
                    s["contents"], dict(toc="", footer=s.footer, header=s.header)
                ),
                axis=1,
            )
            .gather()
        )
        self.update(html.rename("html"))

    async def template_index(self):
        pass

    async def export(self):
        populated = self.dropna(subset="html")
        await Series(
            vectorize(write_file_with_directory)(populated.target, populated.html)
        ).gather()


#######################################################
# supporting functions used in dataframe manipulations.
#######################################################


def replace_html_attachments(contents):
    self = contents
    if "attachments" in self.cells:
        # this will only operate on markdown cells
        attachments = self.cells.attachments[self.cells.attachments.astype(bool)].dropna()
        self.cells.update(
            self.cells.loc[attachments.index].apply(replace_markdown_cell_attachment, axis=1)
        )


def update_notebooks_with_render_contents(contents):
    self = contents
    for (file, cell), y in self.cells.html.dropna().items():
        metadata = self.loc[file, "contents"]["cells"][cell]["metadata"]
        metadata.setdefault("data", {})["text/html"] = y
    if "text/markdown" in self.bundles:
        for (file, cell, output), html in (
            self.bundles[["text/markdown", "text/html"]]
            .dropna(subset="text/markdown")["text/html"]
            .items()
        ):
            self.loc[file, "contents"]["cells"][cell]["outputs"][output]["data"]["text/html"] = html


def render_markdown_contents(contents):
    self = contents
    if "env" not in self.columns:
        self["env"] = None
    if "html" not in self.cells.columns:
        self.cells["html"] = None

    for file, group in self.cells.groupby(self.cells.index.get_level_values("file")):
        env = dict()
        self.loc[[file], "env"] = [env]
        self.cells.update(group.assign(html=group.apply(render_markdown_string, env=env, axis=1)))

        try:
            outs = self.bundles.loc[[file]]
        except KeyError:
            continue
        if "text/markdown" in self.bundles:
            data = (
                outs["text/markdown"]
                .dropna()
                .apply(get_markdown_it_renderer().render, env=env)
                .to_frame("text/html")
            )
            if "text/html" not in self.bundles:
                self.bundles["text/html"] = None
            self.bundles.update(data)
    return self


def collect_contents_headings_title(contents) -> Contents:
    self = contents
    self.headings = (
        concat(
            [
                self.bundles["text/html"].dropna().apply(bs4.BeautifulSoup, features="lxml"),
                (this := self.cells.html.dropna())
                .apply(bs4.BeautifulSoup, features="lxml")
                .to_frame("x")
                .set_index(Index([-1] * len(this), name="output"), append=True),
            ],
            axis=0,
        )
        .x.dropna()
        .pipe(get_contents_headings)
        .sort_index()
    )
    self["title"] = None
    self.update(
        self.headings.groupby(self.headings.index.get_level_values("file"))
        .apply(lambda x: x.iloc[0].string)
        .rename("title")
    )
    return self


def collect_contents_headings(self):
    """extract headings from cell and mimebundle html representations"""
    return (
        concat(
            [
                self.bundles["text/html"].dropna().apply(bs4.BeautifulSoup, features="lxml"),
                (this := self.cells.html.dropna())
                .apply(bs4.BeautifulSoup, features="lxml")
                .to_frame("x")
                .set_index(Index([-1] * len(this), name="output"), append=True),
            ],
            axis=0,
        )
        .x.dropna()
        .pipe(get_contents_headings)
        .sort_index()
    )


def collect_site_navigation(self):
    if "header" not in self.index:
        self["header"] = self["footer"] = None
    distances = DataFrame(
        vectorize(compute_relative_path)(self.target.values[:, None], self.target.values[None, :]),
        self.index,
        self.index,
    ).loc[self.index]
    self.update(distances.apply(get_site_navigation, index=self, axis=1).rename("header"))
    self.update(
        self.apply(
            lambda s: (
                s.prev
                and f"""<a href="{distances.loc[s.name, s.prev]}" rel="prev><span aria-hidden="true">&lt;</span>{self.loc[s.prev].title}</a><br/>"""
                or ""
            )
            + (
                s.next
                and f"""<a href="{distances.loc[s.name, s.next]}" rel="next">{self.loc[s.next].title} <span aria-hidden="true">&gt;</span></a><br/>"""
                or ""
            ),
            axis=1,
        ).rename("footer")
    )


def get_site_navigation(series, index):
    return """<details><summary id="docs-nav">site navigation</summary>
        <ul aria-labelledby="docs-nav">%s</ul></details>""" % "\n".join(
        f"""<li><a href="{rel}">{index.loc[id].title}</a></li>""" for id, rel in series.items()
    )


def get_contents_headings(soup):
    h = soup.methodcaller("select", "h1,h2,h3,h4,h5,h6")
    h = h.enumerate("h").dropna()
    return h.to_frame("h").assign(
        level=h.attrgetter("name").str.lstrip("h").astype(int),
        string=h.attrgetter("text").str.rstrip("¶"),
        id=h.attrgetter("attrs").itemgetter("id"),
        href=h.attrgetter("attrs").itemgetter("href"),
    )

    async def get_execute(self, config, index):
        pass


def prepare_notebook_contents(nb):
    """make inplace changes to the notebook that carried through the publishing process"""
    for cell in nb["cells"]:
        # ensure the source code is a string
        cell.source = "".join(cell.source)
        # hide midgy-ish things
        if MIDGY.match(cell.source):
            cell.metadata.setdefault("jupyter", {})["source_hidden"] = True
        # ensure string outputs are string outputs
        for out in cell.get("outputs", ""):
            for k, v in out.get("data", {}).items():
                k.startswith("text") and out["data"].__setitem__(k, "".join(v))
            if "text" in out:
                out.text = "".join(out.text)


@lru_cache(1)
def get_markdown_it_renderer():
    import markdown_it
    from mdit_py_plugins import anchors, deflist, tasklists

    return (
        markdown_it.MarkdownIt("gfm-like")
        .use(anchors.anchors_plugin)
        .use(deflist.deflist_plugin)
        .use(tasklists.tasklists_plugin)
    )


def render_markdown_string(series, env=None):
    if series.cell_type == "markdown":
        return get_markdown_it_renderer().render(series.source, env=env)
    return None


def compute_relative_path(source, target):
    """compute a relative path from source to target"""
    import pathlib

    if target:
        common = []
        # make both of the paths absolute
        if not source.is_absolute():
            source = pathlib.Path(source).absolute()
        if not target.is_absolute():
            target = pathlib.Path(target).absolute()
        # find all the common parts
        for common, (s, t) in enumerate(zip(source.parts, target.parts)):
            if s != t:
                break  # when the parts are not the same.
        # return the original target type that pops from the source
        # to the common directory then reference the rest of the target parts.
        # the difference in relative paths is NOT symmetric
        return type(target)(*[".."] * (len(target.parents) - common), *target.parts[common:])


@lru_cache(1)
def get_notebook_exporter() -> nbconvert.Exporter:
    """return an nbconvert exporter to translate notebooks to other formats."""
    return nbconvert.get_exporter("a11y")(
        exclude_input_prompt=True,
        include_sa11y=True,
        exclude_output_prompt=True,
        hide_anchor_links=True,
        include_settings=True,
        exclude_anchor_links=True,
        embed_images=True,
        validate_nb=False,
        include_visibility=True,
    )


# this approach has a lot of bloat and could be streamlined later
async def export_from_notebook_node(
    nb: nbformat.NotebookNode, resources: dict = None, exporter: "nbconvert.Exporter" = None
):
    """asynced version of nbconvert exporting"""
    return (exporter or get_notebook_exporter()).from_notebook_node(nb, resources=resources)[0]


async def write_file_with_directory(file: Path, content: str) -> None:
    """write a file ensuring its directory exists"""
    await file.parent.mkdir(exist_ok=True, parents=True)
    await file.write_text(content)


def replace_markdown_cell_attachment(series: Series) -> Series:
    for image, payload in series.loc["attachments"].items():
        for type, data in payload.items():
            series.loc["html"] = series.loc["html"].replace(
                f"attachment:{image}", f"data:{type};base64,{data}"
            )
    return series


def set_target_files(self):
    """establish the next and previous urls to visit in the content stack"""
    if "target" not in self.index:
        self["target"] = self["prev"] = self["next"] = None
    target = self.config.target / self.index.apath.relative_to(self.config.root)
    self.update(target.apath.with_suffix(".html").rename("target"))
    self.update(Series(self.index[1:].tolist() + [None], self.index, name="prev"))
    self.update(Series([None] + self.index[:-1].tolist(), self.index, name="next"))
