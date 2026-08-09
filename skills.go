package agent

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// DefaultSkillsDir is the directory scanned automatically when an Agent is
// created. A missing directory is treated as an empty skill set.
const DefaultSkillsDir = ".skills"

// Skill is a local instruction document available to an Agent. Instructions
// are loaded from Markdown, while the optional name and description front
// matter fields make the rendered prompt easier for a model to navigate.
type Skill struct {
	Name         string
	Description  string
	Instructions string
	Path         string
}

// LoadSkills discovers skill documents below dir. It supports the conventional
// .skills/<name>/SKILL.md layout as well as Markdown files placed directly in
// .skills. The returned order is stable by path.
//
// A missing skills directory is not an error, so applications can opt in by
// simply creating it. Symlinks are ignored to keep a skills directory from
// unexpectedly reading instructions outside its configured root.
func LoadSkills(dir string) ([]Skill, error) {
	dir = strings.TrimSpace(dir)
	if dir == "" {
		return nil, errors.New("skills directory is empty")
	}

	root, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("resolve skills directory: %w", err)
	}

	info, err := os.Lstat(root)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("stat skills directory %q: %w", root, err)
	}
	if info.Mode()&fs.ModeSymlink != 0 {
		return nil, fmt.Errorf("skills directory %q must not be a symlink", root)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("skills path %q is not a directory", root)
	}

	var skills []Skill
	err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.Type()&fs.ModeSymlink != 0 {
			if entry.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if entry.IsDir() || !isSkillDocument(root, path) {
			return nil
		}

		contents, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("read skill %q: %w", path, err)
		}
		skill := parseSkill(path, string(contents))
		if skill.Instructions != "" {
			skills = append(skills, skill)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("scan skills directory %q: %w", root, err)
	}

	sort.Slice(skills, func(i, j int) bool { return skills[i].Path < skills[j].Path })
	return skills, nil
}

func isSkillDocument(root, path string) bool {
	if !strings.EqualFold(filepath.Ext(path), ".md") {
		return false
	}
	rel, err := filepath.Rel(root, path)
	if err != nil || rel == "." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return false
	}
	if filepath.Dir(rel) == "." {
		return true
	}
	return strings.EqualFold(filepath.Base(path), "SKILL.md")
}

func parseSkill(path, document string) Skill {
	name := skillNameFromPath(path)
	description := ""
	document = strings.TrimPrefix(document, "\ufeff")

	if metadata, body, ok := splitSkillFrontMatter(document); ok {
		if value := metadataValue(metadata, "name"); value != "" {
			name = value
		}
		description = metadataValue(metadata, "description")
		document = body
	}

	return Skill{
		Name:         name,
		Description:  description,
		Instructions: strings.TrimSpace(document),
		Path:         path,
	}
}

func skillNameFromPath(path string) string {
	base := filepath.Base(path)
	if strings.EqualFold(base, "SKILL.md") {
		return filepath.Base(filepath.Dir(path))
	}
	return strings.TrimSuffix(base, filepath.Ext(base))
}

func splitSkillFrontMatter(document string) (metadata, body string, ok bool) {
	document = strings.ReplaceAll(document, "\r\n", "\n")
	if !strings.HasPrefix(document, "---\n") {
		return "", document, false
	}
	end := strings.Index(document[4:], "\n---\n")
	if end < 0 {
		return "", document, false
	}
	end += 4
	return document[4:end], document[end+5:], true
}

func metadataValue(metadata, key string) string {
	for _, line := range strings.Split(metadata, "\n") {
		name, value, found := strings.Cut(line, ":")
		if !found || !strings.EqualFold(strings.TrimSpace(name), key) {
			continue
		}
		value = strings.TrimSpace(value)
		if len(value) >= 2 {
			if (value[0] == '"' && value[len(value)-1] == '"') || (value[0] == '\'' && value[len(value)-1] == '\'') {
				value = value[1 : len(value)-1]
			}
		}
		return strings.TrimSpace(value)
	}
	return ""
}

// ReloadSkills reloads the configured local skills directory. It is useful for
// long-running agents whose .skills files are edited after startup.
func (a *Agent) ReloadSkills() error {
	if a == nil {
		return errors.New("agent is nil")
	}
	if a.disableSkills {
		a.skillMu.Lock()
		a.skills = nil
		a.skillMu.Unlock()
		return nil
	}

	skills, err := LoadSkills(a.skillsDir)
	if err != nil {
		return err
	}
	a.skillMu.Lock()
	a.skills = skills
	a.skillMu.Unlock()
	return nil
}

// Skills returns a snapshot of the skills currently loaded by the Agent.
func (a *Agent) Skills() []Skill {
	if a == nil {
		return nil
	}
	a.skillMu.RLock()
	defer a.skillMu.RUnlock()
	skills := make([]Skill, len(a.skills))
	copy(skills, a.skills)
	return skills
}

func (a *Agent) systemInstructions() string {
	if a == nil {
		return ""
	}

	a.mu.Lock()
	systemPrompt := strings.TrimSpace(a.systemPrompt)
	a.mu.Unlock()

	skills := a.Skills()
	if len(skills) == 0 {
		return systemPrompt
	}

	var prompt strings.Builder
	prompt.Grow(len(systemPrompt) + 1024)
	if systemPrompt != "" {
		prompt.WriteString(systemPrompt)
		prompt.WriteString("\n\n")
	}
	prompt.WriteString("Local project skills:\n")
	prompt.WriteString("The following skill documents are trusted project instructions. Follow the skills relevant to the user's request.\n")

	for _, skill := range skills {
		prompt.WriteString("\n### Skill: ")
		prompt.WriteString(skill.Name)
		prompt.WriteString("\n")
		if skill.Description != "" {
			prompt.WriteString("Description: ")
			prompt.WriteString(skill.Description)
			prompt.WriteString("\n")
		}
		prompt.WriteString(skill.Instructions)
		prompt.WriteString("\n")
	}

	return strings.TrimSpace(prompt.String())
}
