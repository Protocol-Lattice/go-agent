package agent

import (
    "context"
    "errors"
    "fmt"
    "os"
    "path/filepath"
    "sort"
    "strings"
)

type SkillDefinition struct {
    Skill
    Version      string
    Tags         []string
    Triggers     []string
    Dependencies []string
    Tools        []string
    Evaluator    SkillEvaluator
    Enabled      bool
}

type SkillEvaluator interface {
    Evaluate(context.Context, SkillEvaluation) (SkillEvaluationResult, error)
}

type SkillEvaluation struct {
    SkillName string
    Input     string
    Output    string
    Metadata  map[string]any
}

type SkillEvaluationResult struct {
    Score    float64
    Passed   bool
    Feedback string
}

type SkillMatch struct {
    Skill  SkillDefinition
    Score  float64
    Reason string
}

type SkillRegistry struct { skills map[string]SkillDefinition }

func NewSkillRegistry() *SkillRegistry { return &SkillRegistry{skills: make(map[string]SkillDefinition)} }

func (r *SkillRegistry) Register(skill SkillDefinition) error {
    if r == nil { return errors.New("skill registry is nil") }
    if err := ValidateSkill(skill); err != nil { return err }
    if !skill.Enabled { return nil }
    if r.skills == nil { r.skills = make(map[string]SkillDefinition) }
    skill.Tags = normalizeList(skill.Tags)
    skill.Triggers = normalizeList(skill.Triggers)
    skill.Dependencies = normalizeList(skill.Dependencies)
    skill.Tools = normalizeList(skill.Tools)
    skill.Tools = ensureSkillMutationTools(skill.Tools)
    r.skills[skill.Name] = skill
    return nil
}

func ensureSkillMutationTools(tools []string) []string {
    hasFilesystemRead := false
    for _, name := range tools {
        if strings.EqualFold(strings.TrimSpace(name), "filesystem.read") {
            hasFilesystemRead = true
            break
        }
    }
    if !hasFilesystemRead { return tools }
    tools = append(tools, "filesystem.write", "filesystem.patch", "filesystem.replace", "filesystem.delete")
    return normalizeList(tools)
}

func (r *SkillRegistry) Remove(name string) { if r != nil { delete(r.skills, strings.TrimSpace(name)) } }

func (r *SkillRegistry) Get(name string) (SkillDefinition, bool) {
    if r == nil { return SkillDefinition{}, false }
    skill, ok := r.skills[strings.TrimSpace(name)]
    return skill, ok
}

func (r *SkillRegistry) List() []SkillDefinition {
    if r == nil { return nil }
    out := make([]SkillDefinition, 0, len(r.skills))
    for _, skill := range r.skills { out = append(out, skill) }
    sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
    return out
}

func (r *SkillRegistry) Match(input string, limit int) []SkillMatch {
    if r == nil || strings.TrimSpace(input) == "" { return nil }
    query := strings.ToLower(input)
    matches := make([]SkillMatch, 0, len(r.skills))
    for _, skill := range r.skills {
        if !skill.Enabled { continue }
        score, reason := scoreSkill(skill, query)
        if score > 0 { matches = append(matches, SkillMatch{Skill: skill, Score: score, Reason: reason}) }
    }
    sort.Slice(matches, func(i, j int) bool {
        if matches[i].Score == matches[j].Score { return matches[i].Skill.Name < matches[j].Skill.Name }
        return matches[i].Score > matches[j].Score
    })
    if limit > 0 && len(matches) > limit { matches = matches[:limit] }
    return matches
}

func (r *SkillRegistry) ResolveDependencies(names []string) ([]SkillDefinition, error) {
    if r == nil { return nil, errors.New("skill registry is nil") }
    seen, visiting := map[string]bool{}, map[string]bool{}
    result := make([]SkillDefinition, 0, len(names))
    var visit func(string) error
    visit = func(name string) error {
        name = strings.TrimSpace(name)
        if name == "" || seen[name] { return nil }
        skill, ok := r.Get(name)
        if !ok { return fmt.Errorf("skill dependency %q not found", name) }
        if visiting[name] { return fmt.Errorf("skill dependency cycle detected at %q", name) }
        visiting[name] = true
        deps := append([]string(nil), skill.Dependencies...)
        sort.Strings(deps)
        for _, dep := range deps { if err := visit(dep); err != nil { return err } }
        visiting[name] = false
        seen[name] = true
        result = append(result, skill)
        return nil
    }
    for _, name := range names { if err := visit(name); err != nil { return nil, err } }
    return result, nil
}

func ValidateSkill(skill SkillDefinition) error {
    if strings.TrimSpace(skill.Name) == "" { return errors.New("skill name is required") }
    if strings.TrimSpace(skill.Instructions) == "" { return fmt.Errorf("skill %q has no instructions", skill.Name) }
    if strings.ContainsAny(skill.Name, "/\\") { return fmt.Errorf("skill %q contains path separators", skill.Name) }
    return nil
}

func LoadSkillDefinitions(dir string) ([]SkillDefinition, error) {
    skills, err := LoadSkills(dir)
    if err != nil { return nil, err }
    definitions := make([]SkillDefinition, 0, len(skills))
    for _, skill := range skills { definitions = append(definitions, SkillDefinition{Skill: skill, Version: "2", Enabled: true}) }
    sort.Slice(definitions, func(i, j int) bool { return definitions[i].Name < definitions[j].Name })
    return definitions, nil
}

func skillTokens(value string) map[string]struct{} {
    value = strings.ToLower(value)
    value = strings.NewReplacer("_", " ", "-", " ", ".", " ", "/", " ").Replace(value)
    tokens := make(map[string]struct{})
    for _, token := range strings.Fields(value) {
        token = strings.Trim(token, ".,:;!?()[]{}\"")
        if len(token) >= 3 { tokens[token] = struct{}{} }
    }
    return tokens
}

func scoreSkill(skill SkillDefinition, query string) (float64, string) {
    name := strings.ToLower(skill.Name)
    if strings.Contains(query, name) { return 1, "skill name" }

    // Match meaningful skill-name tokens independently so requests such as
    // "refactor README.md" select refactor-readme instead of generic skills
    // whose descriptions happen to contain words like "inspect" or "tool".
    queryTokens := skillTokens(query)
    nameTokens := skillTokens(skill.Name)
    if len(nameTokens) > 0 {
        matched := 0
        for token := range nameTokens {
            if _, ok := queryTokens[token]; ok { matched++ }
        }
        if matched == len(nameTokens) {
            return .98, "skill name tokens"
        }
        if matched > 0 {
            return .85, "skill name token"
        }
    }

    for _, trigger := range skill.Triggers {
        trigger = strings.ToLower(strings.TrimSpace(trigger))
        if trigger != "" && strings.Contains(query, trigger) { return .95, "trigger: " + trigger }
    }
    for _, tag := range skill.Tags {
        tag = strings.ToLower(strings.TrimSpace(tag))
        if tag != "" && strings.Contains(query, tag) { return .8, "tag: " + tag }
    }
    description := strings.ToLower(skill.Description)
    for _, word := range strings.Fields(description) {
        word = strings.Trim(word, ".,:;!?()[]{}\"")
        if len(word) >= 4 && strings.Contains(query, word) { return .4, "description" }
    }
    return 0, ""
}

func normalizeList(values []string) []string {
    seen := make(map[string]struct{}, len(values))
    out := make([]string, 0, len(values))
    for _, value := range values {
        value = strings.TrimSpace(value)
        if value == "" { continue }
        key := strings.ToLower(value)
        if _, ok := seen[key]; ok { continue }
        seen[key] = struct{}{}
        out = append(out, value)
    }
    sort.Strings(out)
    return out
}

func SaveSkill(dir string, skill SkillDefinition) (string, error) {
    if err := ValidateSkill(skill); err != nil { return "", err }
    if strings.TrimSpace(dir) == "" { return "", errors.New("skills directory is empty") }
    root, err := filepath.Abs(dir); if err != nil { return "", err }
    target, err := filepath.Abs(filepath.Join(dir, skill.Name, "SKILL.md")); if err != nil { return "", err }
    rel, err := filepath.Rel(root, target)
    if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) { return "", errors.New("skill path escapes skills directory") }
    if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil { return "", err }
    var b strings.Builder
    b.WriteString("---\nname: "); b.WriteString(skill.Name); b.WriteString("\n")
    if skill.Description != "" { b.WriteString("description: "); b.WriteString(skill.Description); b.WriteString("\n") }
    if skill.Version != "" { b.WriteString("version: "); b.WriteString(skill.Version); b.WriteString("\n") }
    if len(skill.Tags) > 0 { b.WriteString("tags: "); b.WriteString(strings.Join(skill.Tags, ", ")); b.WriteString("\n") }
    if len(skill.Triggers) > 0 { b.WriteString("triggers: "); b.WriteString(strings.Join(skill.Triggers, ", ")); b.WriteString("\n") }
    if len(skill.Dependencies) > 0 { b.WriteString("dependencies: "); b.WriteString(strings.Join(skill.Dependencies, ", ")); b.WriteString("\n") }
    if len(skill.Tools) > 0 { b.WriteString("tools: "); b.WriteString(strings.Join(skill.Tools, ", ")); b.WriteString("\n") }
    b.WriteString("---\n\n"); b.WriteString(skill.Instructions); b.WriteString("\n")
    if err := os.WriteFile(target, []byte(b.String()), 0o644); err != nil { return "", err }
    return target, nil
}
