---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
list_title: 'Posts'
title: Welcome
date: 2021-06-27
---

<div class="hero-section">
  <div class="d-flex align-items-center mb-4" style="gap: 1.5rem; flex-wrap: wrap;">
    <img src="/{{ site.profile_path }}" alt="André Fernandes" class="hero-avatar">
    <div>
      <h1 class="hero-name mb-1">André Fernandes</h1>
      <p class="hero-role mb-2">{{ site.data.navbar.subtitle }}</p>
      <p class="hero-description mb-0">{{ site.data.navbar.description }}</p>
    </div>
  </div>

  <div class="hero-body mt-3">
    <p>Welcome to my personal website.</p>

    <p>My name is André Fernandes and I'm a <strong>Portuguese software engineer</strong> with a MSc in Informatics and Computer Engineering at the <strong>Faculty of Engineering of the University of Porto</strong>.</p>

    <p>I come from <strong>Vila Nova de Famalicão</strong>, a city located in the <strong>Braga</strong> district. I'm currently living in <strong>Porto</strong>.</p>

    <p>My main interests are <strong>software development</strong>, <strong>computer science</strong>, <strong>technology</strong>, <strong>horror movies</strong>, <strong>sports</strong> and <strong>travelling</strong>.</p>

    <p>In here, you'll find <strong>professional information</strong> about myself as well as some tech/computer-science related posts that I enjoy writing about and some things I've learned through my work/study experience.</p>

    <p>Feel free to <strong>contact me</strong> through the provided social media.</p>
  </div>

  <div class="hero-cta mt-4" style="display: flex; gap: 0.75rem; flex-wrap: wrap;">
    <a href="/resume/" class="btn btn-accent">View Resume</a>
    <a href="/posts/" class="btn btn-outline-secondary" style="border-radius: 8px; font-weight: 500; padding: 0.6rem 1.4rem;">Read Posts</a>
  </div>
</div>
