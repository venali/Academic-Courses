   Practical Deep Learning for Coders - 4: Natural Language (NLP) code{white-space: pre-wrap;} span.smallcaps{font-variant: small-caps;} div.columns{display: flex; gap: min(4vw, 1.5em);} div.column{flex: auto; overflow-x: auto;} div.hanging-indent{margin-left: 1.5em; text-indent: -1.5em;} ul.task-list{list-style: none;} ul.task-list li input\[type="checkbox"\] { width: 0.8em; margin: 0 0.8em 0.2em -1em; /\* quarto-specific, see https://github.com/quarto-dev/quarto-cli/issues/4556 \*/ vertical-align: middle; }        { "location": "navbar", "copy-button": false, "collapse-after": 3, "panel-placement": "end", "type": "overlay", "limit": 50, "keyboard-shortcut": \[ "f", "/", "s" \], "show-item-context": false, "language": { "search-no-results-text": "No results", "search-matching-documents-text": "matching documents", "search-copy-link-title": "Copy link to search", "search-hide-matches-text": "Hide additional matches", "search-more-match-text": "more match in this document", "search-more-matches-text": "more matches in this document", "search-clear-button-title": "Clear", "search-detached-cancel-button-title": "Cancel", "search-submit-button-title": "Submit", "search-label": "Search" } }       

[Practical Deep Learning for Coders](../index.html)

*   [](https://github.com/fastai/course22)

1.  [Part 1](../Lessons/lesson1.html)
2.  [4: Natural Language (NLP)](../Lessons/lesson4.html)

*   [Practical Deep Learning](../index.html)
    
*   Part 1
    
    *   [1: Getting started](../Lessons/lesson1.html)
        
    *   [2: Deployment](../Lessons/lesson2.html)
        
    *   [3: Neural net foundations](../Lessons/lesson3.html)
        
    *   [4: Natural Language (NLP)](../Lessons/lesson4.html)
        
    *   [5: From-scratch model](../Lessons/lesson5.html)
        
    *   [6: Random forests](../Lessons/lesson6.html)
        
    *   [7: Collaborative filtering](../Lessons/lesson7.html)
        
    *   [8: Convolutions (CNNs)](../Lessons/lesson8.html)
        
    *   [Bonus: Data ethics](../Lessons/lesson8a.html)
        
    *   Summaries
        
        *   [Lesson 1](../Lessons/Summaries/lesson1.html)
            
        *   [Lesson 2](../Lessons/Summaries/lesson2.html)
            
        *   [Lesson 3](../Lessons/Summaries/lesson3.html)
            
        *   [Lesson 4](../Lessons/Summaries/lesson4.html)
            
        *   [Lesson 5](../Lessons/Summaries/lesson5.html)
            
        *   [Lesson 6](../Lessons/Summaries/lesson6.html)
            
        *   [Lesson 7](../Lessons/Summaries/lesson7.html)
            
        *   [Lesson 8](../Lessons/Summaries/lesson8.html)
            
*   Part 2
    
    *   [Part 2 overview](../Lessons/part2.html)
        
    *   [9: Stable Diffusion](../Lessons/lesson9.html)
        
    *   [10: Diving Deeper](../Lessons/lesson10.html)
        
    *   [11: Matrix multiplication](../Lessons/lesson11.html)
        
    *   [12: Mean shift clustering](../Lessons/lesson12.html)
        
    *   [13: Backpropagation & MLP](../Lessons/lesson13.html)
        
    *   [14: Backpropagation](../Lessons/lesson14.html)
        
    *   [15: Autoencoders](../Lessons/lesson15.html)
        
    *   [16: The Learner framework](../Lessons/lesson16.html)
        
    *   [17: Initialization/normalization](../Lessons/lesson17.html)
        
    *   [18: Accelerated SGD & ResNets](../Lessons/lesson18.html)
        
    *   [19: DDPM and Dropout](../Lessons/lesson19.html)
        
    *   [20: Mixed Precision](../Lessons/lesson20.html)
        
    *   [21: DDIM](../Lessons/lesson21.html)
        
    *   [22: Karras et al (2022)](../Lessons/lesson22.html)
        
    *   [23: Super-resolution](../Lessons/lesson23.html)
        
    *   [24: Attention & transformers](../Lessons/lesson24.html)
        
    *   [25: Latent diffusion](../Lessons/lesson25.html)
        
    *   [Bonus: Lesson 9a](https://youtu.be/0_BBRNYInx8)
        
    *   [Bonus: Lesson 9b](https://youtu.be/mYpjmM7O-30)
        
*   Resources
    
    *   [The book](../Resources/book.html)
        
    *   [Forums](../Resources/forums.html)
        
    *   [Kaggle](../Resources/kaggle.html)
        
    *   [Testimonials](../Resources/testimonials.html)
        

## On this page

*   [Video](#video)
*   [Resources](#resources)

*   [Report an issue](https://github.com/fastai/course22-web/issues/new)

1.  [Part 1](../Lessons/lesson1.html)
2.  [4: Natural Language (NLP)](../Lessons/lesson4.html)

# 4: Natural Language (NLP)

![](../images/cute_bunny.png)

It’s time for us to learn how to analyse natural language documents, using Natural Language Processing (NLP). We’ll be focusing on the [Hugging Face](https://huggingface.co/) ecosystem, especially the [Transformers](https://huggingface.co/docs/transformers/index) library, and the vast collection of pretrained [NLP models](https://huggingface.co/models). Our project today will be to classify that similarity of phrases used to describe [US patents](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching). A similar approach can be applied to a wide variety of practical issues, in fields as wide-reaching as marketing, logistics, and medicine.

## Video

This lesson is based partly on [chapter 10](https://github.com/fastai/fastbook/blob/master/10_nlp.ipynb) of the [book](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527).

## Resources

*   Notebook: [Getting started with NLP for absolute beginners](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners)

window.document.addEventListener("DOMContentLoaded", function (event) { const toggleBodyColorMode = (bsSheetEl) => { const mode = bsSheetEl.getAttribute("data-mode"); const bodyEl = window.document.querySelector("body"); if (mode === "dark") { bodyEl.classList.add("quarto-dark"); bodyEl.classList.remove("quarto-light"); } else { bodyEl.classList.add("quarto-light"); bodyEl.classList.remove("quarto-dark"); } } const toggleBodyColorPrimary = () => { const bsSheetEl = window.document.querySelector("link#quarto-bootstrap"); if (bsSheetEl) { toggleBodyColorMode(bsSheetEl); } } toggleBodyColorPrimary(); const icon = ""; const anchorJS = new window.AnchorJS(); anchorJS.options = { placement: 'right', icon: icon }; anchorJS.add('.anchored'); const isCodeAnnotation = (el) => { for (const clz of el.classList) { if (clz.startsWith('code-annotation-')) { return true; } } return false; } const clipboard = new window.ClipboardJS('.code-copy-button', { text: function(trigger) { const codeEl = trigger.previousElementSibling.cloneNode(true); for (const childEl of codeEl.children) { if (isCodeAnnotation(childEl)) { childEl.remove(); } } return codeEl.innerText; } }); clipboard.on('success', function(e) { // button target const button = e.trigger; // don't keep focus button.blur(); // flash "checked" button.classList.add('code-copy-button-checked'); var currentTitle = button.getAttribute("title"); button.setAttribute("title", "Copied!"); let tooltip; if (window.bootstrap) { button.setAttribute("data-bs-toggle", "tooltip"); button.setAttribute("data-bs-placement", "left"); button.setAttribute("data-bs-title", "Copied!"); tooltip = new bootstrap.Tooltip(button, { trigger: "manual", customClass: "code-copy-button-tooltip", offset: \[0, -8\]}); tooltip.show(); } setTimeout(function() { if (tooltip) { tooltip.hide(); button.removeAttribute("data-bs-title"); button.removeAttribute("data-bs-toggle"); button.removeAttribute("data-bs-placement"); } button.setAttribute("title", currentTitle); button.classList.remove('code-copy-button-checked'); }, 1000); // clear code selection e.clearSelection(); }); function tippyHover(el, contentFn, onTriggerFn, onUntriggerFn) { const config = { allowHTML: true, maxWidth: 500, delay: 100, arrow: false, appendTo: function(el) { return el.parentElement; }, interactive: true, interactiveBorder: 10, theme: 'quarto', placement: 'bottom-start', }; if (contentFn) { config.content = contentFn; } if (onTriggerFn) { config.onTrigger = onTriggerFn; } if (onUntriggerFn) { config.onUntrigger = onUntriggerFn; } window.tippy(el, config); } const noterefs = window.document.querySelectorAll('a\[role="doc-noteref"\]'); for (var i=0; i<noterefs.length; i++) { const ref = noterefs\[i\]; tippyHover(ref, function() { // use id or data attribute instead here let href = ref.getAttribute('data-footnote-href') || ref.getAttribute('href'); try { href = new URL(href).hash; } catch {} const id = href.replace(/^#\\/?/, ""); const note = window.document.getElementById(id); return note.innerHTML; }); } const xrefs = window.document.querySelectorAll('a.quarto-xref'); const processXRef = (id, note) => { // Strip column container classes const stripColumnClz = (el) => { el.classList.remove("page-full", "page-columns"); if (el.children) { for (const child of el.children) { stripColumnClz(child); } } } stripColumnClz(note) const typesetMath = (el) => { if (window.MathJax) { // MathJax Typeset window.MathJax.typeset(\[el\]); } else if (window.katex) { // KaTeX Render var mathElements = el.getElementsByClassName("math"); var macros = \[\]; for (var i = 0; i < mathElements.length; i++) { var texText = mathElements\[i\].firstChild; if (mathElements\[i\].tagName == "SPAN") { window.katex.render(texText.data, mathElements\[i\], { displayMode: mathElements\[i\].classList.contains('display'), throwOnError: false, macros: macros, fleqn: false }); } } } } if (id === null || id.startsWith('sec-')) { // Special case sections, only their first couple elements const container = document.createElement("div"); if (note.children && note.children.length > 2) { for (let i = 0; i < 2; i++) { container.appendChild(note.children\[i\].cloneNode(true)); } typesetMath(container); return container.innerHTML } else { typesetMath(note); return note.innerHTML; } } else { // Remove any anchor links if they are present const anchorLink = note.querySelector('a.anchorjs-link'); if (anchorLink) { anchorLink.remove(); } typesetMath(note); return note.innerHTML; } } for (var i=0; i<xrefs.length; i++) { const xref = xrefs\[i\]; tippyHover(xref, undefined, function(instance) { instance.disable(); let url = xref.getAttribute('href'); let hash = undefined; if (url.startsWith('#')) { hash = url; } else { try { hash = new URL(url).hash; } catch {} } if (hash) { const id = hash.replace(/^#\\/?/, ""); const note = window.document.getElementById(id); if (note !== null) { try { const html = processXRef(id, note.cloneNode(true)); instance.setContent(html); } finally { instance.enable(); instance.show(); } } else { // See if we can fetch this fetch(url.split('#')\[0\]) .then(res => res.text()) .then(html => { const parser = new DOMParser(); const htmlDoc = parser.parseFromString(html, "text/html"); const note = htmlDoc.getElementById(id); if (note !== null) { const html = processXRef(id, note); instance.setContent(html); } }).finally(() => { instance.enable(); instance.show(); }); } } else { // See if we can fetch a full url (with no hash to target) // This is a special case and we should probably do some content thinning / targeting fetch(url) .then(res => res.text()) .then(html => { const parser = new DOMParser(); const htmlDoc = parser.parseFromString(html, "text/html"); const note = htmlDoc.querySelector('main.content'); if (note !== null) { // This should only happen for chapter cross references // (since there is no id in the URL) // remove the first header if (note.children.length > 0 && note.children\[0\].tagName === "HEADER") { note.children\[0\].remove(); } const html = processXRef(null, note); instance.setContent(html); } }).finally(() => { instance.enable(); instance.show(); }); } }, function(instance) { }); } let selectedAnnoteEl; const selectorForAnnotation = ( cell, annotation) => { let cellAttr = 'data-code-cell="' + cell + '"'; let lineAttr = 'data-code-annotation="' + annotation + '"'; const selector = 'span\[' + cellAttr + '\]\[' + lineAttr + '\]'; return selector; } const selectCodeLines = (annoteEl) => { const doc = window.document; const targetCell = annoteEl.getAttribute("data-target-cell"); const targetAnnotation = annoteEl.getAttribute("data-target-annotation"); const annoteSpan = window.document.querySelector(selectorForAnnotation(targetCell, targetAnnotation)); const lines = annoteSpan.getAttribute("data-code-lines").split(","); const lineIds = lines.map((line) => { return targetCell + "-" + line; }) let top = null; let height = null; let parent = null; if (lineIds.length > 0) { //compute the position of the single el (top and bottom and make a div) const el = window.document.getElementById(lineIds\[0\]); top = el.offsetTop; height = el.offsetHeight; parent = el.parentElement.parentElement; if (lineIds.length > 1) { const lastEl = window.document.getElementById(lineIds\[lineIds.length - 1\]); const bottom = lastEl.offsetTop + lastEl.offsetHeight; height = bottom - top; } if (top !== null && height !== null && parent !== null) { // cook up a div (if necessary) and position it let div = window.document.getElementById("code-annotation-line-highlight"); if (div === null) { div = window.document.createElement("div"); div.setAttribute("id", "code-annotation-line-highlight"); div.style.position = 'absolute'; parent.appendChild(div); } div.style.top = top - 2 + "px"; div.style.height = height + 4 + "px"; div.style.left = 0; let gutterDiv = window.document.getElementById("code-annotation-line-highlight-gutter"); if (gutterDiv === null) { gutterDiv = window.document.createElement("div"); gutterDiv.setAttribute("id", "code-annotation-line-highlight-gutter"); gutterDiv.style.position = 'absolute'; const codeCell = window.document.getElementById(targetCell); const gutter = codeCell.querySelector('.code-annotation-gutter'); gutter.appendChild(gutterDiv); } gutterDiv.style.top = top - 2 + "px"; gutterDiv.style.height = height + 4 + "px"; } selectedAnnoteEl = annoteEl; } }; const unselectCodeLines = () => { const elementsIds = \["code-annotation-line-highlight", "code-annotation-line-highlight-gutter"\]; elementsIds.forEach((elId) => { const div = window.document.getElementById(elId); if (div) { div.remove(); } }); selectedAnnoteEl = undefined; }; // Handle positioning of the toggle window.addEventListener( "resize", throttle(() => { elRect = undefined; if (selectedAnnoteEl) { selectCodeLines(selectedAnnoteEl); } }, 10) ); function throttle(fn, ms) { let throttle = false; let timer; return (...args) => { if(!throttle) { // first call gets through fn.apply(this, args); throttle = true; } else { // all the others get throttled if(timer) clearTimeout(timer); // cancel #2 timer = setTimeout(() => { fn.apply(this, args); timer = throttle = false; }, ms); } }; } // Attach click handler to the DT const annoteDls = window.document.querySelectorAll('dt\[data-target-cell\]'); for (const annoteDlNode of annoteDls) { annoteDlNode.addEventListener('click', (event) => { const clickedEl = event.target; if (clickedEl !== selectedAnnoteEl) { unselectCodeLines(); const activeEl = window.document.querySelector('dt\[data-target-cell\].code-annotation-active'); if (activeEl) { activeEl.classList.remove('code-annotation-active'); } selectCodeLines(clickedEl); clickedEl.classList.add('code-annotation-active'); } else { // Unselect the line unselectCodeLines(); clickedEl.classList.remove('code-annotation-active'); } }); } const findCites = (el) => { const parentEl = el.parentElement; if (parentEl) { const cites = parentEl.dataset.cites; if (cites) { return { el, cites: cites.split(' ') }; } else { return findCites(el.parentElement) } } else { return undefined; } }; var bibliorefs = window.document.querySelectorAll('a\[role="doc-biblioref"\]'); for (var i=0; i<bibliorefs.length; i++) { const ref = bibliorefs\[i\]; const citeInfo = findCites(ref); if (citeInfo) { tippyHover(citeInfo.el, function() { var popup = window.document.createElement('div'); citeInfo.cites.forEach(function(cite) { var citeDiv = window.document.createElement('div'); citeDiv.classList.add('hanging-indent'); citeDiv.classList.add('csl-entry'); var biblioDiv = window.document.getElementById('ref-' + cite); if (biblioDiv) { citeDiv.innerHTML = biblioDiv.innerHTML; } popup.appendChild(citeDiv); }); return popup.innerHTML; }); } } });

[3: Neural net foundations](../Lessons/lesson3.html)

[5: From-scratch model](../Lessons/lesson5.html)
